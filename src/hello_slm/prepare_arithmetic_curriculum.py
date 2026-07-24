from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hello_slm.config import canonical_json_bytes, file_sha256, repo_root

SPLITS = ("train", "validation", "test")
DEFAULT_CREATED_AT = "2026-07-22T00:00:00Z"
DEFAULT_OUTPUT_DIR = Path("data/arithmetic-curriculum")
DEFAULT_LOCK_PATH = Path("data/sources/arithmetic-curriculum.lock.json")
SOURCE_NAME = "hello-slm-arithmetic-curriculum"
LICENSE = "MIT"
RIGHTS_HOLDER = "Self-authored synthetic"


class CurriculumError(ValueError):
    """Raised when arithmetic curriculum generation cannot continue."""


@dataclass(frozen=True)
class Fact:
    operation: str
    left: int
    right: int
    expected_answer: int

    @property
    def key(self) -> str:
        return f"{self.operation}:{self.left}:{self.right}:{self.expected_answer}"


@dataclass(frozen=True)
class CurriculumRecord:
    conversation_id: str
    question: str
    answer: str
    operation: str
    left: int
    right: int
    expected_answer: int
    variant: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic Hello SLM arithmetic curriculum."
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-seed", type=int, default=3101)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    args = parser.parse_args(argv)

    try:
        report = prepare(
            lock_path=args.lock,
            output_dir=args.output_dir,
            split_seed=args.split_seed,
            split_ratios={
                "train": args.train_ratio,
                "validation": args.validation_ratio,
                "test": args.test_ratio,
            },
        )
    except CurriculumError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"status": "success", **report["generated"]}, sort_keys=True))
    return 0


def prepare(
    *,
    lock_path: Path = DEFAULT_LOCK_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    split_seed: int = 3101,
    split_ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    ratios = split_ratios or {"train": 0.98, "validation": 0.01, "test": 0.01}
    _validate_ratios(ratios)
    lock = _read_json(lock_path)
    records = list(generate_curriculum_records())
    _validate_records(records)
    split_rows = split_records(records, split_seed=split_seed, split_ratios=ratios)

    created_at = str(lock.get("generator", {}).get("created_at", DEFAULT_CREATED_AT))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = _write_split_files(output_dir, split_rows, created_at)
    report = _build_report(lock, records, split_rows, split_seed, ratios)
    _atomic_write_json(output_dir / "preparation-report.json", report)
    manifest = {
        "format_version": 1,
        "name": SOURCE_NAME,
        "created_at": created_at,
        "entries": manifest_entries,
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return report


def generate_curriculum_records(
    *,
    add_limit: int = 100,
    subtract_limit: int = 100,
    multiply_limit: int = 50,
    division_quotient_limit: int = 50,
    division_divisor_limit: int = 50,
) -> Iterator[CurriculumRecord]:
    for fact in generate_facts(
        add_limit=add_limit,
        subtract_limit=subtract_limit,
        multiply_limit=multiply_limit,
        division_quotient_limit=division_quotient_limit,
        division_divisor_limit=division_divisor_limit,
    ):
        for variant in (0, 1):
            question, answer = _render_fact(fact, variant)
            yield CurriculumRecord(
                conversation_id=_conversation_id(question, answer),
                question=question,
                answer=answer,
                operation=fact.operation,
                left=fact.left,
                right=fact.right,
                expected_answer=fact.expected_answer,
                variant=variant,
            )


def generate_facts(
    *,
    add_limit: int = 100,
    subtract_limit: int = 100,
    multiply_limit: int = 50,
    division_quotient_limit: int = 50,
    division_divisor_limit: int = 50,
) -> Iterator[Fact]:
    for left in range(add_limit):
        for right in range(add_limit):
            yield Fact("add", left, right, left + right)
    for left in range(subtract_limit):
        for right in range(subtract_limit):
            yield Fact("subtract", left, right, left - right)
    for left in range(multiply_limit):
        for right in range(multiply_limit):
            yield Fact("multiply", left, right, left * right)
    for quotient in range(division_quotient_limit):
        for divisor in range(1, division_divisor_limit + 1):
            yield Fact("divide", quotient * divisor, divisor, quotient)


def split_records(
    records: Iterable[CurriculumRecord],
    *,
    split_seed: int,
    split_ratios: dict[str, float],
) -> dict[str, list[CurriculumRecord]]:
    train_boundary = split_ratios["train"]
    validation_boundary = train_boundary + split_ratios["validation"]
    result: dict[str, list[CurriculumRecord]] = {split: [] for split in SPLITS}
    for record in records:
        digest = hashlib.sha256(f"{split_seed}\n{record.conversation_id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        split = (
            "train"
            if bucket < train_boundary
            else "validation"
            if bucket < validation_boundary
            else "test"
        )
        result[split].append(record)
    return result


def _render_fact(fact: Fact, variant: int) -> tuple[str, str]:
    if fact.operation == "add":
        symbol = "+"
        word = "plus"
        first_question = f"Add {fact.left} and {fact.right}."
        second_question = f"What is {fact.left} + {fact.right}?"
    elif fact.operation == "subtract":
        symbol = "-"
        word = "minus"
        first_question = f"Subtract {fact.right} from {fact.left}."
        second_question = f"What is {fact.left} - {fact.right}?"
    elif fact.operation == "multiply":
        symbol = "*"
        word = "times"
        first_question = f"Multiply {fact.left} by {fact.right}."
        second_question = f"What is {fact.left} * {fact.right}?"
    elif fact.operation == "divide":
        symbol = "/"
        word = "divided by"
        first_question = f"Divide {fact.left} by {fact.right}."
        second_question = f"What is {fact.left} / {fact.right}?"
    else:
        raise CurriculumError(f"unknown operation {fact.operation!r}")

    if variant == 0:
        return (
            first_question,
            f"{fact.left} {symbol} {fact.right} = {fact.expected_answer}.",
        )
    if variant == 1:
        return (
            second_question,
            (
                f"For {fact.operation} fact {fact.left}-{fact.right}, "
                f"{fact.left} {word} {fact.right} equals {fact.expected_answer}."
            ),
        )
    raise CurriculumError(f"unknown variant {variant!r}")


def _conversation_id(question: str, answer: str) -> str:
    digest = hashlib.sha256(f"{question}\n---answer---\n{answer}".encode()).hexdigest()
    return f"arith.curr.{digest[:24]}"


def _validate_records(records: list[CurriculumRecord]) -> None:
    if len(records) != 50_000:
        raise CurriculumError(f"expected 50000 records, got {len(records)}")
    counters = Counter(record.operation for record in records)
    expected = {"add": 20_000, "subtract": 20_000, "multiply": 5_000, "divide": 5_000}
    if counters != expected:
        raise CurriculumError(f"unexpected operation counts: {dict(counters)}")
    _ensure_unique("conversation_id", [record.conversation_id for record in records])
    _ensure_unique("question", [record.question for record in records])
    _ensure_unique("assistant target", [record.answer for record in records])


def _ensure_unique(name: str, values: list[str]) -> None:
    if len(set(values)) != len(values):
        raise CurriculumError(f"{name} values must be unique")


def _validate_ratios(ratios: dict[str, float]) -> None:
    if set(ratios) != set(SPLITS):
        raise CurriculumError("split ratios must define train, validation, and test")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise CurriculumError(f"split ratios must sum to 1.0, got {total}")
    if any(value < 0 for value in ratios.values()):
        raise CurriculumError("split ratios must not be negative")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CurriculumError(f"could not read lock file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CurriculumError(f"lock file is not valid JSON: {exc}") from exc


def _write_split_files(
    output_dir: Path, split_rows: dict[str, list[CurriculumRecord]], created_at: str
) -> list[dict[str, Any]]:
    entries = []
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        rows = [
            _conversation_record(record, created_at)
            for record in sorted(split_rows[split], key=lambda item: item.conversation_id)
        ]
        _atomic_write_jsonl(path, rows)
        entries.append(
            {
                "path": _relative_manifest_path(path),
                "split": split,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "conversation_count": len(rows),
                "provenance": "Self-authored deterministic synthetic arithmetic curriculum.",
                "license": LICENSE,
                "rights_holder": RIGHTS_HOLDER,
                "consent": {
                    "training_allowed": True,
                    "granted_by": RIGHTS_HOLDER,
                    "grant_reference": "Self-authored MIT synthetic dataset lock",
                },
                "allowed_use": _allowed_use(split),
                "contains_personal_data": False,
                "contains_synthetic_data": True,
                "included": True,
            }
        )
    return entries


def _conversation_record(record: CurriculumRecord, created_at: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "conversation_id": record.conversation_id,
        "source": SOURCE_NAME,
        "license": LICENSE,
        "created_at": created_at,
        "messages": [
            {"role": "user", "content": record.question},
            {"role": "assistant", "content": record.answer},
        ],
        "metadata": {
            "operation": record.operation,
            "left": record.left,
            "right": record.right,
            "expected_answer": record.expected_answer,
            "variant": record.variant,
        },
    }


def _allowed_use(split: str) -> list[str]:
    if split == "train":
        return ["tokenizer-training", "model-training"]
    if split == "validation":
        return ["validation", "evaluation"]
    return ["test", "evaluation"]


def _relative_manifest_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        parts = path.resolve().parts
        if "data" in parts:
            data_index = len(parts) - 1 - parts[::-1].index("data")
            return Path(*parts[data_index:]).as_posix()
        return path.name


def _build_report(
    lock: dict[str, Any],
    records: list[CurriculumRecord],
    split_rows: dict[str, list[CurriculumRecord]],
    split_seed: int,
    split_ratios: dict[str, float],
) -> dict[str, Any]:
    operation_counts = Counter(record.operation for record in records)
    variant_counts = Counter(str(record.variant) for record in records)
    split_counts = {split: len(split_rows[split]) for split in SPLITS}
    return {
        "format_version": 1,
        "source_lock": lock,
        "generated": {
            "total": len(records),
            "splits": split_counts,
            "operations": dict(sorted(operation_counts.items())),
            "variants": dict(sorted(variant_counts.items())),
        },
        "split_seed": split_seed,
        "split_ratios": split_ratios,
        "near_duplicate_mode": "exact_only",
        "deduplication": {
            "conversation_ids_unique": True,
            "questions_unique": True,
            "assistant_targets_unique": True,
        },
    }


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    _atomic_write_bytes(path, raw)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with open(handle, "wb") as output:
            output.write(raw)
            output.flush()
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
