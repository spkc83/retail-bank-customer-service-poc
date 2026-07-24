from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hello_slm.config import canonical_json_bytes, file_sha256, repo_root

SPLITS = ("train", "validation", "test")
DEFAULT_CREATED_AT = "2026-07-22T00:00:00Z"
SOURCE_NAME = "orca-math-word-problems-200k"
REPLACEMENTS = {
    "\u00a0": " ",
    "\u00b0": " degrees ",
    "\u00b7": "*",
    "\u00d7": "*",
    "\u00f7": "/",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": ",",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2026": "...",
    "\u2032": "'",
    "\u2033": '"',
    "\u2044": "/",
    "\u20ac": "EUR",
    "\u2212": "-",
    "\u2217": "*",
    "\u2219": "*",
    "\u221a": "sqrt",
    "\u221e": "infinity",
    "\u2248": "approximately",
    "\u2260": "!=",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u03c0": "pi",
}


class PreparationError(ValueError):
    """Raised when math corpus preparation cannot continue."""


@dataclass(frozen=True)
class Candidate:
    conversation_id: str
    question: str
    answer: str
    source_row: int
    source_digest: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Orca Math arithmetic conversations.")
    parser.add_argument("--lock", type=Path, default=Path("data/sources/orca-math.lock.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/arithmetic"))
    parser.add_argument("--max-records", type=int, default=50_000)
    parser.add_argument("--split-seed", type=int, default=2101)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = prepare(
            lock_path=args.lock,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            max_records=args.max_records,
            split_seed=args.split_seed,
            split_ratios={
                "train": args.train_ratio,
                "validation": args.validation_ratio,
                "test": args.test_ratio,
            },
            force_download=args.force_download,
        )
    except PreparationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"status": "success", **report["selected"]}, sort_keys=True))
    return 0


def prepare(
    *,
    lock_path: Path,
    raw_dir: Path,
    output_dir: Path,
    max_records: int,
    split_seed: int,
    split_ratios: dict[str, float],
    force_download: bool = False,
) -> dict[str, Any]:
    if max_records < 1:
        raise PreparationError("--max-records must be at least 1")
    _validate_ratios(split_ratios)
    lock = _read_json(lock_path)
    policy = lock.get("preparation_policy", {})
    source = lock["source"]
    parquet_path = _ensure_source(source, raw_dir, force_download)
    expected_rows = int(source["rows"])

    candidates, scan_report = _scan_parquet(parquet_path, lock, max_records=max_records)
    if scan_report["scanned_rows"] != expected_rows:
        raise PreparationError(
            f"row count mismatch: expected {expected_rows}, got {scan_report['scanned_rows']}"
        )

    selected = sorted(candidates, key=lambda item: (_selection_hash(item), item.conversation_id))[
        :max_records
    ]
    selected = sorted(selected, key=lambda item: item.conversation_id)
    split_rows = _split_candidates(selected, split_seed, split_ratios)

    created_at = str(policy.get("created_at", DEFAULT_CREATED_AT))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = _write_split_files(output_dir, split_rows, created_at)
    report = _build_report(lock, scan_report, split_rows, selected)
    _atomic_write_json(output_dir / "preparation-report.json", report)
    manifest = {
        "format_version": 1,
        "name": "orca-math-arithmetic",
        "created_at": created_at,
        "entries": manifest_entries,
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return report


def _validate_ratios(ratios: dict[str, float]) -> None:
    if set(ratios) != set(SPLITS):
        raise PreparationError("split ratios must define train, validation, and test")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise PreparationError(f"split ratios must sum to 1.0, got {total}")
    if any(value < 0 for value in ratios.values()):
        raise PreparationError("split ratios must not be negative")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PreparationError(f"could not read lock file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PreparationError(f"lock file is not valid JSON: {exc}") from exc


def _ensure_source(source: dict[str, Any], raw_dir: Path, force_download: bool) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = str(source["url"])
    parsed = urllib.parse.urlparse(url)
    filename = Path(urllib.parse.unquote(parsed.path)).name or "source.parquet"
    destination = raw_dir / filename
    if destination.exists() and not force_download:
        _verify_file(destination, source)
        return destination

    tmp = Path(tempfile.mkstemp(prefix=filename + ".", suffix=".tmp", dir=raw_dir)[1])
    try:
        if parsed.scheme == "file":
            source_path = Path(urllib.request.url2pathname(parsed.path))
            if source_path.resolve() == destination.resolve():
                _verify_file(destination, source)
                return destination
            shutil.copyfile(source_path, tmp)
        else:
            with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        _verify_file(tmp, source)
        tmp.replace(destination)
    except OSError as exc:
        raise PreparationError(f"could not acquire source parquet: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
    return destination


def _verify_file(path: Path, source: dict[str, Any]) -> None:
    expected_bytes = int(source["bytes"])
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise PreparationError(
            f"source byte count mismatch for {path}: expected {expected_bytes}, got {actual_bytes}"
        )
    expected_hash = str(source["sha256"])
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise PreparationError(f"source sha256 mismatch for {path}")


def _scan_parquet(
    parquet_path: Path, lock: dict[str, Any], *, max_records: int
) -> tuple[list[Candidate], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as exc:
        raise PreparationError(
            "pyarrow is required to read Orca Math parquet files; install pyarrow to use "
            "hello_slm.prepare_math"
        ) from exc

    policy = lock.get("preparation_policy", {})
    limits = {
        "question": int(policy.get("max_question_chars", 600)),
        "answer": int(policy.get("max_answer_chars", 1200)),
        "combined": int(policy.get("max_combined_chars", 800)),
    }
    parquet = pq.ParquetFile(parquet_path)
    question_col, answer_col = _question_answer_columns(parquet.schema_arrow)

    scanned = 0
    accepted: list[Candidate] = []
    rejections: Counter[str] = Counter()
    replacements: Counter[str] = Counter()
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    char_lengths: dict[str, list[int]] = {"question": [], "answer": [], "combined": []}

    for batch in parquet.iter_batches(columns=[question_col, answer_col], batch_size=2048):
        columns = batch.to_pydict()
        for raw_question, raw_answer in zip(
            columns[question_col], columns[answer_col], strict=True
        ):
            scanned += 1
            question, question_replacements = _normalize(str(raw_question or ""))
            answer, answer_replacements = _normalize(str(raw_answer or ""))
            replacements.update(question_replacements)
            replacements.update(answer_replacements)
            reason = _rejection_reason(question, answer, limits)
            if reason:
                rejections[reason] += 1
                continue

            pair = (question, answer)
            if pair in seen_pairs:
                rejections["duplicate_pair"] += 1
                continue
            if question in seen_questions:
                rejections["duplicate_question"] += 1
                continue
            if answer in seen_answers:
                rejections["duplicate_answer"] += 1
                continue

            seen_pairs.add(pair)
            seen_questions.add(question)
            seen_answers.add(answer)
            char_lengths["question"].append(len(question))
            char_lengths["answer"].append(len(answer))
            char_lengths["combined"].append(len(question) + len(answer))
            accepted.append(_candidate(question, answer, scanned))

    return accepted, {
        "source_path": str(parquet_path),
        "scanned_rows": scanned,
        "accepted_rows": len(accepted),
        "max_records": max_records,
        "rejections": dict(sorted(rejections.items())),
        "replacements": dict(sorted(replacements.items())),
        "char_length_stats": {
            key: _stats(values) for key, values in sorted(char_lengths.items())
        },
    }


def _question_answer_columns(schema: Any) -> tuple[str, str]:
    names = list(schema.names)
    lowered = {name.lower(): name for name in names}
    question = next(
        (
            lowered[key]
            for key in ("question", "problem", "input", "prompt")
            if key in lowered
        ),
        None,
    )
    answer = next(
        (
            lowered[key]
            for key in ("answer", "solution", "output", "response")
            if key in lowered
        ),
        None,
    )
    if question and answer and question != answer:
        return question, answer
    string_names = [
        field.name
        for field in schema
        if str(field.type) in {"string", "large_string"} and field.name in names
    ]
    if len(string_names) >= 2:
        return string_names[0], string_names[1]
    raise PreparationError("could not identify question and answer columns in parquet schema")


def _normalize(value: str) -> tuple[str, Counter[str]]:
    text = unicodedata.normalize("NFKC", value)
    counts: Counter[str] = Counter()
    for source, target in REPLACEMENTS.items():
        count = text.count(source)
        if count:
            counts[source] += count
            text = text.replace(source, target)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    lines = [re.sub(r"[ ]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, counts


def _rejection_reason(question: str, answer: str, limits: dict[str, int]) -> str | None:
    if not question:
        return "empty_question"
    if not answer:
        return "empty_answer"
    if not any(char.isdigit() for char in question + answer):
        return "no_digit"
    if len(question) > limits["question"]:
        return "question_too_long"
    if len(answer) > limits["answer"]:
        return "answer_too_long"
    if len(question) + len(answer) > limits["combined"]:
        return "combined_too_long"
    invalid = [char for char in question + answer if not _ascii_text_char(char)]
    if invalid:
        return "non_ascii_or_control"
    return None


def _ascii_text_char(char: str) -> bool:
    if char == "\n":
        return True
    code = ord(char)
    return 32 <= code <= 126


def _candidate(question: str, answer: str, source_row: int) -> Candidate:
    digest = hashlib.sha256(f"{question}\n---answer---\n{answer}".encode()).hexdigest()
    return Candidate(
        conversation_id=f"orca.math.{digest[:24]}",
        question=question,
        answer=answer,
        source_row=source_row,
        source_digest=digest,
    )


def _selection_hash(candidate: Candidate) -> str:
    return hashlib.sha256(f"select\n{candidate.conversation_id}".encode()).hexdigest()


def _split_candidates(
    candidates: Iterable[Candidate], split_seed: int, ratios: dict[str, float]
) -> dict[str, list[Candidate]]:
    train_boundary = ratios["train"]
    validation_boundary = train_boundary + ratios["validation"]
    result: dict[str, list[Candidate]] = {split: [] for split in SPLITS}
    for candidate in candidates:
        digest = hashlib.sha256(
            f"{split_seed}\n{candidate.conversation_id}".encode()
        ).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        split = (
            "train"
            if bucket < train_boundary
            else "validation"
            if bucket < validation_boundary
            else "test"
        )
        result[split].append(candidate)
    return result


def _write_split_files(
    output_dir: Path, split_rows: dict[str, list[Candidate]], created_at: str
) -> list[dict[str, Any]]:
    entries = []
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        rows = [
            _conversation_record(candidate, created_at)
            for candidate in sorted(split_rows[split], key=lambda item: item.conversation_id)
        ]
        _atomic_write_jsonl(path, rows)
        relative_path = _relative_manifest_path(path)
        entries.append(
            {
                "path": relative_path,
                "split": split,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "conversation_count": len(rows),
                "provenance": (
                    "Prepared from microsoft/orca-math-word-problems-200k pinned parquet."
                ),
                "license": "MIT",
                "rights_holder": "Microsoft",
                "consent": {
                    "training_allowed": True,
                    "granted_by": "Microsoft dataset card",
                    "grant_reference": "MIT license",
                },
                "allowed_use": _allowed_use(split),
                "contains_personal_data": False,
                "contains_synthetic_data": True,
                "included": True,
            }
        )
    return entries


def _conversation_record(candidate: Candidate, created_at: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "conversation_id": candidate.conversation_id,
        "source": SOURCE_NAME,
        "license": "MIT",
        "created_at": created_at,
        "messages": [
            {"role": "user", "content": candidate.question},
            {"role": "assistant", "content": candidate.answer},
        ],
        "metadata": {
            "source_digest": candidate.source_digest,
            "source_row": candidate.source_row,
        },
    }


def _relative_manifest_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        parts = path.resolve().parts
        if "data" in parts:
            data_index = len(parts) - 1 - parts[::-1].index("data")
            return Path(*parts[data_index:]).as_posix()
        return path.name


def _allowed_use(split: str) -> list[str]:
    if split == "train":
        return ["tokenizer-training", "model-training"]
    if split == "validation":
        return ["validation", "evaluation"]
    return ["test", "evaluation"]


def _build_report(
    lock: dict[str, Any],
    scan_report: dict[str, Any],
    split_rows: dict[str, list[Candidate]],
    selected: list[Candidate],
) -> dict[str, Any]:
    split_counts = {split: len(split_rows[split]) for split in SPLITS}
    return {
        "format_version": 1,
        "source_lock": lock,
        "source_path": scan_report["source_path"],
        "scanned_rows": scan_report["scanned_rows"],
        "accepted_rows": scan_report["accepted_rows"],
        "selected": {"total": len(selected), "splits": split_counts},
        "rejections": scan_report["rejections"],
        "rejection_counting": "exclusive_first_match",
        "replacements": scan_report["replacements"],
        "near_duplicate_mode": "exact_only",
        "char_length_stats": scan_report["char_length_stats"],
    }


def _stats(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 4),
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
