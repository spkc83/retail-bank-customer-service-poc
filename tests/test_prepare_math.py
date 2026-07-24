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
from hello_slm.prepare_math import PreparationError, prepare  # noqa: E402

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(path: Path) -> None:
    table = pa.table(
        {
            "question": [
                "What is 2 × 3?",
                "What is 4 plus 5?",
                "What is 6 minus 1?",
                "What is 8 divided by 2?",
                "What is 2 × 3?",
                "What is 7+2?",
                "No arithmetic here",
                "What is 10 + 1?",
                "Bad unicode 3 ☃",
                "What is 12 + 1?",
            ],
            "answer": [
                "2 × 3 = 6.",
                "4 plus 5 equals 9.",
                "6 − 1 = 5.",
                "8 ÷ 2 = 4.",
                "2 × 3 = 6.",
                "4 plus 5 equals 9.",
                "Still no digit.",
                "A" * 1201,
                "3",
                "13",
            ],
        }
    )
    pq.write_table(table, path)


def _lock_for(path: Path) -> dict:
    return {
        "format_version": 1,
        "source": {
            "repo": "microsoft/orca-math-word-problems-200k",
            "revision": "fixture",
            "url": path.resolve().as_uri(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "rows": 10,
            "license": "MIT",
            "rights_holder": "Microsoft",
            "card_labels": {
                "license": "MIT",
                "answers": "synthetic via Azure GPT-4 Turbo",
            },
        },
        "preparation_policy": {
            "created_at": "2026-07-22T00:00:00Z",
            "max_records": 50_000,
            "split_seed": 2101,
            "split_ratios": {"train": 0.98, "validation": 0.01, "test": 0.01},
            "max_question_chars": 600,
            "max_answer_chars": 1200,
            "max_combined_chars": 1500,
            "near_duplicate_mode": "exact_only",
        },
    }


def _prepare_fixture(tmp_path: Path, max_records: int = 4) -> tuple[Path, dict]:
    source = tmp_path / "source.parquet"
    _write_fixture(source)
    lock = _lock_for(source)
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    output_dir = tmp_path / "repo" / "data" / "arithmetic"
    report = prepare(
        lock_path=lock_path,
        raw_dir=tmp_path / "raw",
        output_dir=output_dir,
        max_records=max_records,
        split_seed=2101,
        split_ratios={"train": 0.98, "validation": 0.01, "test": 0.01},
    )
    return output_dir, report


def _jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _expected_split(conversation_id: str) -> str:
    digest = hashlib.sha256(f"2101\n{conversation_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < 0.98:
        return "train"
    if bucket < 0.99:
        return "validation"
    return "test"


def test_prepare_math_is_deterministic_and_loader_compatible(tmp_path: Path) -> None:
    output_dir, report = _prepare_fixture(tmp_path)
    first_bytes = {path.name: path.read_bytes() for path in sorted(output_dir.glob("*"))}

    second_report = prepare(
        lock_path=tmp_path / "lock.json",
        raw_dir=tmp_path / "raw",
        output_dir=output_dir,
        max_records=4,
        split_seed=2101,
        split_ratios={"train": 0.98, "validation": 0.01, "test": 0.01},
    )
    second_bytes = {path.name: path.read_bytes() for path in sorted(output_dir.glob("*"))}

    assert second_report == report
    assert second_bytes == first_bytes
    assert report["scanned_rows"] == 10
    assert report["accepted_rows"] == 5
    assert report["selected"]["total"] == 4
    assert report["rejections"] == {
        "answer_too_long": 1,
        "duplicate_answer": 1,
        "duplicate_pair": 1,
        "no_digit": 1,
        "non_ascii_or_control": 1,
    }
    assert report["rejection_counting"] == "exclusive_first_match"
    assert report["replacements"]["×"] == 4
    assert report["replacements"]["−"] == 1
    assert report["replacements"]["÷"] == 1
    assert report["near_duplicate_mode"] == "exact_only"

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_json_schema(manifest, ROOT / "schemas" / "corpus-manifest.schema.json")
    assert [entry["split"] for entry in manifest["entries"]] == ["train", "validation", "test"]
    for entry in manifest["entries"]:
        path = output_dir / Path(entry["path"]).name
        assert entry["sha256"] == file_sha256(path)
        assert entry["bytes"] == path.stat().st_size
        assert entry["conversation_count"] == len(_jsonl(path))
        assert entry["allowed_use"] == {
            "train": ["tokenizer-training", "model-training"],
            "validation": ["validation", "evaluation"],
            "test": ["test", "evaluation"],
        }[entry["split"]]

    for entry in manifest["entries"]:
        for row in _jsonl(output_dir / Path(entry["path"]).name):
            validate_json_schema(row, ROOT / "schemas" / "conversation.schema.json")
            assert row["conversation_id"].startswith("orca.math.")
            assert entry["split"] == _expected_split(row["conversation_id"])
            assert [message["role"] for message in row["messages"]] == ["user", "assistant"]
            assert "source_digest" in row["metadata"]

    fake_root = tmp_path / "repo"
    (fake_root / "schemas").mkdir()
    for schema in ("conversation.schema.json", "corpus-manifest.schema.json"):
        (fake_root / "schemas" / schema).write_bytes((ROOT / "schemas" / schema).read_bytes())
    config = load_experiment_config(ROOT / "configs" / "arithmetic-30m.toml", ROOT)
    data = copy.deepcopy(config.data)
    data["corpus"]["manifest_path"] = str(output_dir / "manifest.json")
    loaded = load_and_validate_corpus(replace(config, root=fake_root, data=data))
    assert len(loaded.conversations) == 4


def test_prepare_math_rejects_tampered_source(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _write_fixture(source)
    lock = _lock_for(source)
    lock["source"]["sha256"] = "0" * 64
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")

    with pytest.raises(PreparationError, match="source sha256 mismatch"):
        prepare(
            lock_path=lock_path,
            raw_dir=tmp_path / "raw",
            output_dir=tmp_path / "repo" / "data" / "arithmetic",
            max_records=4,
            split_seed=2101,
            split_ratios={"train": 0.98, "validation": 0.01, "test": 0.01},
        )
