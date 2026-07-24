from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hello_slm.config import file_sha256, load_experiment_config, validate_json_schema  # noqa: E402
from hello_slm.data import load_and_validate_corpus  # noqa: E402
from hello_slm.prepare_arithmetic_curriculum import (  # noqa: E402
    CurriculumError,
    generate_curriculum_records,
    prepare,
)

ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _expected_split(conversation_id: str) -> str:
    digest = hashlib.sha256(f"3101\n{conversation_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < 0.98:
        return "train"
    if bucket < 0.99:
        return "validation"
    return "test"


def _assert_arithmetic(row: dict) -> None:
    metadata = row["metadata"]
    left = metadata["left"]
    right = metadata["right"]
    expected = metadata["expected_answer"]
    if metadata["operation"] == "add":
        assert left + right == expected
    elif metadata["operation"] == "subtract":
        assert left - right == expected
    elif metadata["operation"] == "multiply":
        assert left * right == expected
    elif metadata["operation"] == "divide":
        assert right != 0
        assert left / right == expected
        assert left % right == 0
    else:
        raise AssertionError(f"unknown operation {metadata['operation']}")


def test_reduced_generation_counts_uniqueness_and_arithmetic() -> None:
    records = list(
        generate_curriculum_records(
            add_limit=3,
            subtract_limit=3,
            multiply_limit=2,
            division_quotient_limit=2,
            division_divisor_limit=3,
        )
    )

    assert len(records) == 56
    assert sum(record.operation == "add" for record in records) == 18
    assert sum(record.operation == "subtract" for record in records) == 18
    assert sum(record.operation == "multiply" for record in records) == 8
    assert sum(record.operation == "divide" for record in records) == 12
    assert len({record.conversation_id for record in records}) == len(records)
    assert len({record.question for record in records}) == len(records)
    assert len({record.answer for record in records}) == len(records)

    for record in records:
        row = {
            "metadata": {
                "operation": record.operation,
                "left": record.left,
                "right": record.right,
                "expected_answer": record.expected_answer,
            }
        }
        _assert_arithmetic(row)


def test_prepare_curriculum_is_deterministic_schema_valid_and_loader_compatible(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "repo" / "data" / "arithmetic-curriculum"
    lock_path = ROOT / "data" / "sources" / "arithmetic-curriculum.lock.json"
    split_ratios = {"train": 0.98, "validation": 0.01, "test": 0.01}
    report = prepare(
        lock_path=lock_path,
        output_dir=output_dir,
        split_seed=3101,
        split_ratios=split_ratios,
    )
    first_bytes = {path.name: path.read_bytes() for path in sorted(output_dir.glob("*"))}
    second_report = prepare(
        lock_path=lock_path,
        output_dir=output_dir,
        split_seed=3101,
        split_ratios=split_ratios,
    )
    second_bytes = {path.name: path.read_bytes() for path in sorted(output_dir.glob("*"))}

    assert second_report == report
    assert second_bytes == first_bytes
    assert report["generated"]["total"] == 50_000
    assert report["generated"]["operations"] == {
        "add": 20_000,
        "divide": 5_000,
        "multiply": 5_000,
        "subtract": 20_000,
    }
    assert report["generated"]["variants"] == {"0": 25_000, "1": 25_000}
    assert report["near_duplicate_mode"] == "exact_only"
    assert report["deduplication"] == {
        "assistant_targets_unique": True,
        "conversation_ids_unique": True,
        "questions_unique": True,
    }

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_json_schema(manifest, ROOT / "schemas" / "corpus-manifest.schema.json")
    assert [entry["split"] for entry in manifest["entries"]] == ["train", "validation", "test"]
    assert sum(entry["conversation_count"] for entry in manifest["entries"]) == 50_000

    all_rows = []
    for entry in manifest["entries"]:
        path = output_dir / Path(entry["path"]).name
        rows = _jsonl(path)
        all_rows.extend(rows)
        assert entry["sha256"] == file_sha256(path)
        assert entry["bytes"] == path.stat().st_size
        assert entry["conversation_count"] == len(rows)
        assert entry["license"] == "MIT"
        assert entry["rights_holder"] == "Self-authored synthetic"
        assert entry["contains_personal_data"] is False
        assert entry["contains_synthetic_data"] is True
        assert entry["allowed_use"] == {
            "train": ["tokenizer-training", "model-training"],
            "validation": ["validation", "evaluation"],
            "test": ["test", "evaluation"],
        }[entry["split"]]
        for row in rows:
            validate_json_schema(row, ROOT / "schemas" / "conversation.schema.json")
            assert row["source"] == "hello-slm-arithmetic-curriculum"
            assert row["license"] == "MIT"
            assert [message["role"] for message in row["messages"]] == ["user", "assistant"]
            assert entry["split"] == _expected_split(row["conversation_id"])
            _assert_arithmetic(row)

    assert len({row["conversation_id"] for row in all_rows}) == 50_000
    assert len({row["messages"][0]["content"] for row in all_rows}) == 50_000
    assert len({row["messages"][1]["content"] for row in all_rows}) == 50_000

    fake_root = tmp_path / "repo"
    (fake_root / "schemas").mkdir()
    for schema in ("conversation.schema.json", "corpus-manifest.schema.json"):
        (fake_root / "schemas" / schema).write_bytes((ROOT / "schemas" / schema).read_bytes())
    config = load_experiment_config(ROOT / "configs" / "arithmetic-30m.toml", ROOT)
    data = copy.deepcopy(config.data)
    data["corpus"]["manifest_path"] = str(output_dir / "manifest.json")
    data["seeds"]["corpus_split"] = 3101
    loaded = load_and_validate_corpus(replace(config, root=fake_root, data=data))
    assert len(loaded.conversations) == 50_000
    assert loaded.report["near_duplicate_similarity_checked"] is False


def test_prepare_curriculum_rejects_invalid_ratios(tmp_path: Path) -> None:
    with pytest.raises(CurriculumError, match="sum to 1.0"):
        prepare(
            lock_path=ROOT / "data" / "sources" / "arithmetic-curriculum.lock.json",
            output_dir=tmp_path / "out",
            split_seed=3101,
            split_ratios={"train": 0.9, "validation": 0.2, "test": 0.0},
        )
