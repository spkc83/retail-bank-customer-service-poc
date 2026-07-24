from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hello_slm.config import canonical_json_bytes, file_sha256

SPLITS = ("train", "validation", "test")
CREATED_AT = "2026-07-24T00:00:00Z"
SOURCE_NAME = "hello-slm-banking-v2"
SYSTEM_PROMPT = (
    "You are a retail banking support assistant. Help with accounts, cards, transfers, "
    "payments, loans, fees, branches, ATMs, and related financial-services support. "
    "If the user asks about another domain, give the standard out-of-domain response."
)
BANKING_V2_CANNED_OOD_RESPONSE = (
    "I can only help with retail banking and financial-services questions. Please ask about "
    "accounts, cards, transfers, payments, loans, or related banking support."
)
BITEXT_DATASET_ID = "bitext/Bitext-retail-banking-llm-chatbot-training-dataset"
BITEXT_REVISION = "3e3621092fc6baaf7f53ceb6f091c60ae99acb67"
BITEXT_LICENSE = "CDLA-Sharing-1.0"
BITEXT_CSV = "bitext-retail-banking-llm-chatbot-training-dataset.csv"
POLYAI_DATASET_ID = "PolyAI/banking77"
POLYAI_REVISION = "90d4e2ee5521c04fc1488f065b8b083658768c57"
POLYAI_LICENSE = "CC-BY-4.0"
DEFAULT_SOURCE_DIR = Path("data/sources")
DEFAULT_OUTPUT_DIR = Path("data/banking-v2")
DEFAULT_LOCK_PATH = Path("data/sources/banking-v2.lock.json")
DEFAULT_BITEXT_SNAPSHOT = Path("data/sources/banking-v2-bitext-source.jsonl")
DEFAULT_POLYAI_SNAPSHOT = Path("data/sources/banking-v2-polyai-banking77-eval-source.jsonl")
PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
ORPHAN_TRAILING_PLACEHOLDER_RE = re.compile(r"\b([A-Z][A-Za-z ]{2,80})\}\}")
ORPHAN_LEADING_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Z][A-Za-z ]{2,80})\b")
WHITESPACE_RE = re.compile(r"[ \t]+")
NON_WORD_RE = re.compile(r"[^a-z0-9]+")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
LONG_NUMBER_RE = re.compile(r"\b(?:\d[ -]?){12,19}\b")
PHONE_RE = re.compile(r"\b(?:\+?1[ -.]?)?(?:\(?\d{3}\)?[ -.]?)\d{3}[ -.]\d{4}\b")

PLACEHOLDER_VALUES = {
    "American Express Customer Support Phone Number": "American Express customer support",
    "American Express Customer Support Working Hours": "published support hours",
    "Bank Account": "bank account",
    "Bank Branch": "local branch",
    "Bank Name": "your bank",
    "Banking App": "mobile banking app",
    "Card Services": "card services",
    "Company Website URL": "the bank website",
    "Credit Card": "credit card",
    "Customer Service Phone Number": "customer service",
    "Customer Support Phone Number": "customer support",
    "Customer Support Working Hours": "published support hours",
    "Live Chat": "secure chat",
    "Manage Cards": "manage cards",
    "Name": "name",
    "Account Details": "account details",
    "Password": "password",
    "Username": "username",
}


class BankingDataError(ValueError):
    """Raised when banking-v2 source preparation cannot continue."""


@dataclass(frozen=True)
class SourceSnapshot:
    path: Path
    dataset_id: str
    revision: str
    license: str
    sha256: str


@dataclass(frozen=True)
class PreparedRecord:
    conversation_id: str
    split_key: str
    source: str
    license: str
    messages: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]
    forced_split: str | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the banking-v2 SFT corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit-sources")
    audit_parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    audit_parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--bitext-snapshot", type=Path, default=DEFAULT_BITEXT_SNAPSHOT)
    prepare_parser.add_argument("--polyai-snapshot", type=Path, default=DEFAULT_POLYAI_SNAPSHOT)
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    prepare_parser.add_argument("--split-seed", type=int, default=7101)

    args = parser.parse_args(argv)
    try:
        if args.command == "audit-sources":
            lock = audit_sources(source_dir=args.source_dir, lock_path=args.lock)
            print(json.dumps({"status": "success", "lock": str(args.lock), **lock["summary"]}))
        elif args.command == "prepare":
            lock = _read_json(args.lock)
            snapshot = SourceSnapshot(
                path=args.bitext_snapshot,
                dataset_id=BITEXT_DATASET_ID,
                revision=str(lock["sources"][BITEXT_DATASET_ID]["revision"]),
                license=BITEXT_LICENSE,
                sha256=str(lock["sources"][BITEXT_DATASET_ID]["snapshot_sha256"]),
            )
            report = prepare(
                snapshot=snapshot,
                output_dir=args.output_dir,
                lock_path=args.lock,
                split_seed=args.split_seed,
                polyai_snapshot=args.polyai_snapshot,
            )
            print(json.dumps({"status": "success", **report["summary"]}, sort_keys=True))
    except (BankingDataError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def audit_sources(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> dict[str, Any]:
    source_dir.mkdir(parents=True, exist_ok=True)
    bitext_csv = source_dir / "banking-v2-bitext-source.csv"
    bitext_snapshot = source_dir / DEFAULT_BITEXT_SNAPSHOT.name
    polyai_snapshot = source_dir / DEFAULT_POLYAI_SNAPSHOT.name

    _download(
        _hf_resolve_url(BITEXT_DATASET_ID, BITEXT_REVISION, BITEXT_CSV),
        bitext_csv,
    )
    bitext_rows = _read_csv(bitext_csv)
    _write_source_snapshot(
        bitext_snapshot,
        (
            {
                "source_row_id": index,
                "tags": str(row.get("tags", "")),
                "instruction": str(row.get("instruction", "")),
                "category": str(row.get("category", "")),
                "intent": str(row.get("intent", "")),
                "response": str(row.get("response", "")),
            }
            for index, row in enumerate(bitext_rows)
        ),
    )

    polyai_rows = list(_load_polyai_rows())
    _write_source_snapshot(polyai_snapshot, polyai_rows)

    lock = _build_lock(
        bitext_csv=bitext_csv,
        bitext_snapshot=bitext_snapshot,
        polyai_snapshot=polyai_snapshot,
        polyai_rows=len(polyai_rows),
    )
    _atomic_write_json(lock_path, lock)
    return lock


def prepare(
    *,
    snapshot: SourceSnapshot,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    lock_path: Path = DEFAULT_LOCK_PATH,
    split_seed: int = 7101,
    polyai_snapshot: Path | None = None,
) -> dict[str, Any]:
    bitext_records = list(_records_from_bitext(snapshot, split_seed=split_seed))
    synthetic_records = list(_synthetic_records(split_seed=split_seed))
    records = bitext_records + synthetic_records
    split_rows = _split_records(records, split_seed=split_seed)
    router_eval_rows = _router_eval_rows(polyai_snapshot)
    report = _build_report(
        records=records,
        split_rows=split_rows,
        router_eval_rows=router_eval_rows,
        snapshot=snapshot,
        lock_path=lock_path,
        polyai_snapshot=polyai_snapshot,
        split_seed=split_seed,
    )
    _validate_report(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = _write_split_files(output_dir, split_rows)
    router_entry = _write_router_eval_file(output_dir, router_eval_rows)
    manifest = {
        "format_version": 2,
        "name": SOURCE_NAME,
        "created_at": CREATED_AT,
        "contract": "banking-v2-manifest",
        "generative_sft": manifest_entries,
        "router_eval": [router_entry],
        "policy": {
            "allowed_trainable_licenses": ["CDLA-Sharing-1.0", "MIT"],
            "allowed_router_eval_licenses": [POLYAI_LICENSE],
            "duplicate_assistant_targets": "allowed only when metadata.task == 'ood_gate'",
        },
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    _atomic_write_json(output_dir / "preparation-report.json", report)
    (output_dir / "DATA_CARD.md").write_text(_render_data_card(report), encoding="utf-8")
    return report


def validate_banking_v2_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    required = {"format_version", "name", "created_at", "contract", "generative_sft", "router_eval"}
    missing = required - set(manifest)
    if missing:
        raise BankingDataError(f"manifest missing fields: {sorted(missing)}")
    if manifest["format_version"] != 2 or manifest["contract"] != "banking-v2-manifest":
        raise BankingDataError("not a banking-v2 manifest")
    base = path.parent
    assistant_targets: dict[str, list[str]] = defaultdict(list)
    for entry in manifest["generative_sft"]:
        rows = _read_jsonl(base / Path(entry["path"]).name)
        if len(rows) != int(entry["conversation_count"]):
            raise BankingDataError(f"{entry['name']} conversation_count mismatch")
        for row in rows:
            metadata = row.get("metadata", {})
            if metadata.get("trainable") is not True:
                raise BankingDataError("generative SFT row is not marked trainable")
            if row.get("source") == "PolyAI/banking77":
                raise BankingDataError("Banking77 row entered generative SFT")
            if "{{" in json.dumps(row) or "}}" in json.dumps(row):
                raise BankingDataError("unresolved placeholder in manifest row")
            for message in row["messages"]:
                if count_pii_like(message["content"]):
                    raise BankingDataError("PII-like text remains in manifest row")
                if message["role"] == "assistant":
                    assistant_targets[message["content"]].append(str(metadata.get("task")))
    for target, tasks in assistant_targets.items():
        if len(tasks) > 1 and set(tasks) != {"ood_gate"}:
            raise BankingDataError(f"duplicate assistant target is not OOD-only: {target}")
    for entry in manifest["router_eval"]:
        rows = _read_jsonl(base / Path(entry["path"]).name)
        if len(rows) != int(entry["conversation_count"]):
            raise BankingDataError("router eval conversation_count mismatch")
        for row in rows:
            if row.get("source") != "PolyAI/banking77":
                raise BankingDataError("router eval row is not Banking77")
            if row["metadata"].get("trainable") is not False:
                raise BankingDataError("Banking77 router eval row is trainable")
            for message in row["messages"]:
                if count_pii_like(message["content"]):
                    raise BankingDataError("PII-like text remains in router eval row")
    return manifest


def normalize_bitext_placeholders(text: str) -> tuple[str, int]:
    count = 0
    text = text.replace("{{{{", "{{").replace("}}}}", "}}")

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        name = " ".join(match.group(1).split())
        return PLACEHOLDER_VALUES.get(name, _humanize_placeholder(name))

    normalized = PLACEHOLDER_RE.sub(replace, text)
    normalized = ORPHAN_TRAILING_PLACEHOLDER_RE.sub(replace, normalized)
    normalized = ORPHAN_LEADING_PLACEHOLDER_RE.sub(replace, normalized)
    normalized = _normalize_text(normalized)
    return normalized, count


def scrub_pii_like(text: str) -> tuple[str, int]:
    replacements = 0
    patterns = (
        (EMAIL_RE, "redacted email address"),
        (SSN_RE, "redacted taxpayer identifier"),
        (PHONE_RE, "redacted phone number"),
        (LONG_NUMBER_RE, "redacted long number"),
    )
    scrubbed = text
    for pattern, replacement in patterns:
        scrubbed, count = pattern.subn(replacement, scrubbed)
        replacements += count
    return _normalize_text(scrubbed), replacements


def count_pii_like(text: str) -> int:
    patterns = (EMAIL_RE, SSN_RE, PHONE_RE, LONG_NUMBER_RE)
    return sum(len(pattern.findall(text)) for pattern in patterns)


def _records_from_bitext(snapshot: SourceSnapshot, *, split_seed: int) -> Iterator[PreparedRecord]:
    raw_rows = []
    seen_instruction_keys: set[str] = set()
    for row in _read_source_snapshot(snapshot.path):
        instruction = _required_text(row, "instruction")
        response = _required_text(row, "response")
        category = _required_text(row, "category").upper()
        intent = _required_text(row, "intent")
        tags = str(row.get("tags", ""))
        source_row_id = int(row.get("source_row_id", 0))
        instruction, instruction_placeholders = normalize_bitext_placeholders(instruction)
        response, response_placeholders = normalize_bitext_placeholders(response)
        instruction, instruction_pii = scrub_pii_like(instruction)
        response, response_pii = scrub_pii_like(response)
        if "{{" in instruction or "}}" in instruction or "{{" in response or "}}" in response:
            raise BankingDataError(f"unresolved placeholder in Bitext row {source_row_id}")
        instruction_key = _normalized_user_key(instruction)
        if not instruction_key:
            raise BankingDataError(
                f"instruction is empty after normalization in Bitext row {source_row_id}"
            )
        if instruction_key in seen_instruction_keys:
            continue
        seen_instruction_keys.add(instruction_key)
        raw_rows.append(
            {
                "source_row_id": source_row_id,
                "tags": tags,
                "instruction": instruction,
                "response": response,
                "category": category,
                "intent": intent,
                "instruction_key": instruction_key,
                "placeholder_replacements": instruction_placeholders + response_placeholders,
                "pii_replacements": instruction_pii + response_pii,
            }
        )
    clusters = _cluster_bitext_rows(raw_rows)
    for item in raw_rows:
        source_row_id_value = int(str(item["source_row_id"]))
        cluster_id = clusters[source_row_id_value]
        split_key = f"bitext:{item['category']}:{item['intent']}:cluster-{cluster_id}"
        yield PreparedRecord(
            conversation_id=_conversation_id(
                "bitext",
                str(item["source_row_id"]),
                str(item["instruction"]),
                str(item["response"]),
            ),
            split_key=split_key,
            source="bitext-retail-banking-llm-chatbot-training-dataset",
            license=snapshot.license,
            messages=(
                _message("system", SYSTEM_PROMPT, loss=False),
                _message("user", str(item["instruction"]), loss=False),
                _message("assistant", str(item["response"]), loss=True),
            ),
            metadata={
                "record_type": "bitext_sft",
                "task": "banking_sft",
                "trainable": True,
                "source_dataset": snapshot.dataset_id,
                "source_revision": snapshot.revision,
                "source_row_id": source_row_id_value,
                "category": str(item["category"]),
                "intent": str(item["intent"]),
                "tags": str(item["tags"]),
                "split_group": split_key,
                "bitext_cluster": cluster_id,
                "bitext_cluster_threshold": 0.1,
                "placeholder_replacements": int(str(item["placeholder_replacements"])),
                "pii_replacements": int(str(item["pii_replacements"])),
            },
        )
    _ = split_seed


def _synthetic_records(*, split_seed: int) -> Iterator[PreparedRecord]:
    for split in SPLITS:
        for base in _ood_bases(split):
            split_key = f"synthetic:{split}:ood:{base['topic']}"
            for index, prompt in enumerate(base["prompts"]):
                prompt, pii_replacements = scrub_pii_like(prompt)
                response, response_pii = scrub_pii_like(BANKING_V2_CANNED_OOD_RESPONSE)
                yield PreparedRecord(
                    conversation_id=_conversation_id(
                        "ood", split, base["topic"], str(index), prompt
                    ),
                    split_key=split_key,
                    source="hello-slm-banking-v2-synthetic",
                    license="MIT",
                    messages=(
                        _message("system", SYSTEM_PROMPT, loss=False),
                        _message("user", prompt, loss=False),
                        _message("assistant", response, loss=True),
                    ),
                    metadata={
                        "record_type": "ood_refusal",
                        "task": "ood_gate",
                        "trainable": True,
                        "source_dataset": "self-authored",
                        "source_revision": CREATED_AT,
                        "category": "OOD",
                        "intent": "out_of_domain",
                        "topic": base["topic"],
                        "split_group": split_key,
                        "placeholder_replacements": 0,
                        "pii_replacements": pii_replacements + response_pii,
                    },
                    forced_split=split,
                )

        for base in _multi_turn_bases(split):
            split_key = f"synthetic:{split}:multi:{base['base_issue']}"
            messages = tuple(_message(**message) for message in base["messages"])
            pii_replacements = sum(message.get("pii_replacements", 0) for message in messages)
            yield PreparedRecord(
                conversation_id=_conversation_id(
                    "multi", split, base["base_issue"], base["turn_pattern"]
                ),
                split_key=split_key,
                source="hello-slm-banking-v2-synthetic",
                license="MIT",
                messages=messages,
                metadata={
                    "record_type": "multi_turn",
                    "task": "ood_gate"
                    if base["turn_pattern"] == "in_domain_to_ood"
                    else "banking_sft",
                    "trainable": True,
                    "source_dataset": "self-authored",
                    "source_revision": CREATED_AT,
                    "category": base["category"],
                    "intent": base["intent"],
                    "base_issue": base["base_issue"],
                    "turn_pattern": base["turn_pattern"],
                    "split_group": split_key,
                    "placeholder_replacements": 0,
                    "pii_replacements": pii_replacements,
                },
                forced_split=split,
            )
    _ = split_seed


def _split_records(
    records: Iterable[PreparedRecord], *, split_seed: int
) -> dict[str, list[PreparedRecord]]:
    grouped: dict[str, list[PreparedRecord]] = defaultdict(list)
    forced_rows: dict[str, list[PreparedRecord]] = {split: [] for split in SPLITS}
    for record in records:
        if record.forced_split is not None:
            if record.forced_split not in SPLITS:
                raise BankingDataError(f"unknown forced split {record.forced_split!r}")
            forced_rows[record.forced_split].append(record)
            continue
        grouped[record.split_key].append(record)
    split_by_key = _assign_weighted_split_keys(grouped, split_seed=split_seed)
    split_rows: dict[str, list[PreparedRecord]] = {
        split: sorted(forced_rows[split], key=lambda record: record.conversation_id)
        for split in SPLITS
    }
    for key in sorted(grouped):
        split_rows[split_by_key[key]].extend(
            sorted(grouped[key], key=lambda record: record.conversation_id)
        )
    return split_rows


def _assign_weighted_split_keys(
    grouped: dict[str, list[PreparedRecord]], *, split_seed: int
) -> dict[str, str]:
    """Assign whole dedup groups while approximating a 90/5/5 row split.

    Bitext clusters differ greatly in size, so splitting by number of cluster
    keys can produce a nearly empty held-out set. This deterministic bin
    selection reserves groups closest to each held-out row target, fills any
    remaining capacity with smaller groups, and assigns the rest to training.
    """

    keys = set(grouped)
    if len(keys) < 3:
        ordered = sorted(
            keys,
            key=lambda key: hashlib.sha256(f"{split_seed}\n{key}".encode()).hexdigest(),
        )
        counts = _split_counts(len(ordered))
        assigned: dict[str, str] = {}
        cursor = 0
        for split in SPLITS:
            for key in ordered[cursor : cursor + counts[split]]:
                assigned[key] = split
            cursor += counts[split]
        return assigned

    total_rows = sum(len(rows) for rows in grouped.values())
    heldout_target = max(1, round(total_rows * 0.05))
    assigned = {}
    remaining = set(keys)

    def stable_hash(key: str) -> str:
        return hashlib.sha256(f"{split_seed}\n{key}".encode()).hexdigest()

    for split in ("validation", "test"):
        if len(remaining) <= 1:
            break
        first = min(
            remaining,
            key=lambda key: (abs(len(grouped[key]) - heldout_target), stable_hash(key)),
        )
        assigned[first] = split
        remaining.remove(first)
        selected_rows = len(grouped[first])
        while selected_rows < heldout_target and len(remaining) > 1:
            capacity = heldout_target - selected_rows
            fitting = [key for key in remaining if len(grouped[key]) <= capacity]
            if not fitting:
                break
            key = min(fitting, key=lambda item: (-len(grouped[item]), stable_hash(item)))
            assigned[key] = split
            remaining.remove(key)
            selected_rows += len(grouped[key])

    for key in remaining:
        assigned[key] = "train"
    return assigned


def _split_counts(total_keys: int) -> dict[str, int]:
    if total_keys < 3:
        return {"train": total_keys, "validation": 0, "test": 0}
    validation = max(1, round(total_keys * 0.05))
    test = max(1, round(total_keys * 0.05))
    train = total_keys - validation - test
    if train < 1:
        train = 1
        if validation >= test:
            validation -= 1
        else:
            test -= 1
    return {"train": train, "validation": validation, "test": test}


def _build_report(
    *,
    records: list[PreparedRecord],
    split_rows: dict[str, list[PreparedRecord]],
    router_eval_rows: list[dict[str, Any]],
    snapshot: SourceSnapshot,
    lock_path: Path,
    polyai_snapshot: Path | None,
    split_seed: int,
) -> dict[str, Any]:
    normalized_users_by_split: dict[str, set[str]] = {}
    unresolved = 0
    trainable_quarantined = 0
    remaining_pii_like = 0
    pii_replacements = 0
    for split, rows in split_rows.items():
        user_keys = set()
        for record in rows:
            if not record.metadata["trainable"] and record.metadata["record_type"] != "quarantined":
                trainable_quarantined += 1
            pii_replacements += int(record.metadata.get("pii_replacements", 0))
            for message in record.messages:
                if "{{" in message["content"] or "}}" in message["content"]:
                    unresolved += 1
                remaining_pii_like += count_pii_like(message["content"])
                if message["role"] == "user":
                    user_keys.add(_normalized_user_key(message["content"]))
        normalized_users_by_split[split] = user_keys
    for row in router_eval_rows:
        pii_replacements += int(row["metadata"].get("pii_replacements", 0))
        for message in row["messages"]:
            remaining_pii_like += count_pii_like(message["content"])
    cross_split_duplicates = 0
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            cross_split_duplicates += len(
                normalized_users_by_split[left] & normalized_users_by_split[right]
            )

    canonical = [
        _record_to_json(record, split)
        for split in SPLITS
        for record in sorted(split_rows[split], key=lambda item: item.conversation_id)
    ]
    counts = {
        split: {
            "conversations": len(rows),
            "sources": dict(Counter(record.source for record in rows)),
            "record_types": dict(Counter(str(record.metadata["record_type"]) for record in rows)),
        }
        for split, rows in split_rows.items()
    }
    bitext_records = [
        record for record in records if record.metadata["record_type"] == "bitext_sft"
    ]
    synthetic_coverage = _synthetic_coverage(split_rows)
    return {
        "format_version": 1,
        "name": SOURCE_NAME,
        "created_at": CREATED_AT,
        "summary": {
            "total_conversations": len(records),
            "bitext_sft_conversations": len(bitext_records),
            "synthetic_conversations": len(records) - len(bitext_records),
            "split_seed": split_seed,
            "corpus_fingerprint": hashlib.sha256(canonical_json_bytes(canonical)).hexdigest(),
        },
        "splits": counts,
        "checks": {
            "unresolved_placeholders": unresolved,
            "cross_split_normalized_user_duplicates": cross_split_duplicates,
            "trainable_quarantined_records": trainable_quarantined,
            "banking77_generative_sft_rows": sum(
                1 for record in records if record.source == "PolyAI/banking77"
            ),
            "banking77_router_eval_rows": len(router_eval_rows),
            "ood_duplicate_assistant_targets_allowed_only_for_task": "ood_gate",
            "bitext_split_grouping": "category:intent:near_duplicate_cluster",
            "ood_exact_response_count": sum(
                1 for record in records if record.metadata["record_type"] == "ood_refusal"
            ),
            "pii_like_replacements": pii_replacements,
            "remaining_pii_like_matches": remaining_pii_like,
        },
        "synthetic_coverage": synthetic_coverage,
        "bitext_clustering": {
            "method": "intent-local normalized token 3-gram Jaccard union-find",
            "threshold": 0.1,
            "split_key": "category:intent:cluster",
        },
        "licenses": dict(Counter(record.license for record in records)),
        "source_roles": {
            BITEXT_DATASET_ID: {
                "role": "primary-qa-sft",
                "license": BITEXT_LICENSE,
                "revision": snapshot.revision,
                "snapshot_sha256": snapshot.sha256,
            },
            POLYAI_DATASET_ID: {
                "role": "intent-router-eval-only",
                "license": POLYAI_LICENSE,
                "revision": POLYAI_REVISION,
                "snapshot_path": str(polyai_snapshot) if polyai_snapshot else None,
                "snapshot_sha256": file_sha256(polyai_snapshot)
                if polyai_snapshot and polyai_snapshot.exists()
                else None,
                "trainable": False,
            },
        },
        "ood": {
            "canned_response": BANKING_V2_CANNED_OOD_RESPONSE,
            "split_rule": "topic-level split key ood:<topic>",
        },
        "quarantine": {
            "talkmap": {
                "trainable": False,
                "reason": "Only recorded as explicitly quarantined metadata; no rows are emitted.",
            }
        },
        "excluded_sources": {
            "rakesh": {
                "trainable": False,
                "reason": "Excluded by user requirement; no Rakesh source rows are emitted.",
            }
        },
        "lock_path": str(lock_path),
        "known_integration_gaps": [
            "Existing configs/corpus.toml allows MIT only; banking-v2 needs a separate policy "
            "before load_and_validate_corpus compatibility is claimed.",
            "Existing corpus validator rejects duplicate assistant targets; banking-v2 OOD "
            "requires "
            "many prompts with one exact canned response, so a v2-aware policy is required.",
            "data/banking-v2/ should be added to .gitignore by the integration lane.",
        ],
    }


def _validate_report(report: dict[str, Any]) -> None:
    checks = report["checks"]
    if checks["unresolved_placeholders"] != 0:
        raise BankingDataError("unresolved placeholders remain")
    if checks["cross_split_normalized_user_duplicates"] != 0:
        raise BankingDataError("normalized user text appears in more than one split")
    if checks["trainable_quarantined_records"] != 0:
        raise BankingDataError("quarantined record was emitted as trainable")
    if checks["banking77_generative_sft_rows"] != 0:
        raise BankingDataError("Banking77 rows entered generative SFT")
    if checks["remaining_pii_like_matches"] != 0:
        raise BankingDataError("PII-like text remains after scrubbing")
    required_patterns = {"clarification", "follow_up", "correction", "in_domain_to_ood"}
    for split, coverage in report["synthetic_coverage"].items():
        if coverage["ood_refusal"] < 1:
            raise BankingDataError(f"{split} split has no OOD gate rows")
        missing = required_patterns - set(coverage["multi_turn_patterns"])
        if missing:
            raise BankingDataError(
                f"{split} split is missing multi-turn patterns: {sorted(missing)}"
            )


def _synthetic_coverage(
    split_rows: dict[str, list[PreparedRecord]]
) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for split, rows in split_rows.items():
        patterns = sorted(
            {
                str(record.metadata["turn_pattern"])
                for record in rows
                if record.metadata["record_type"] == "multi_turn"
            }
        )
        coverage[split] = {
            "ood_refusal": sum(
                1 for record in rows if record.metadata["record_type"] == "ood_refusal"
            ),
            "multi_turn": sum(
                1 for record in rows if record.metadata["record_type"] == "multi_turn"
            ),
            "multi_turn_patterns": patterns,
        }
    return coverage


def _write_split_files(
    output_dir: Path, split_rows: dict[str, list[PreparedRecord]]
) -> list[dict[str, Any]]:
    entries = []
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in split_rows[split]:
                handle.write(json.dumps(_record_to_json(record, split), sort_keys=True) + "\n")
        entries.append(
            {
                "name": split,
                "path": str(path),
                "role": "generative_sft",
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "conversation_count": len(split_rows[split]),
                "provenance": (
                    "Bitext retail banking QA plus self-authored banking-v2 "
                    "OOD/multi-turn records"
                ),
                "licenses": ["CDLA-Sharing-1.0", "MIT"],
                "rights_holder": "Bitext and Hello SLM authors",
                "allowed_use": _allowed_use_for_split(split),
                "pii": "none-detected",
                "synthetic_data": True,
                "included_for_generative_sft": True,
            }
        )
    return entries


def _write_router_eval_file(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = output_dir / "banking77-router-eval.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "name": "banking77-router-eval",
        "path": str(path),
        "role": "intent_router_eval",
        "source_dataset": POLYAI_DATASET_ID,
        "license": POLYAI_LICENSE,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "conversation_count": len(rows),
        "included_for_generative_sft": False,
        "pii": "none-detected",
    }


def _router_eval_rows(polyai_snapshot: Path | None) -> list[dict[str, Any]]:
    if polyai_snapshot is None or not polyai_snapshot.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row in _read_source_snapshot(polyai_snapshot):
        text = _required_text(row, "text")
        label = _required_text(row, "label")
        text, pii_replacements = scrub_pii_like(text)
        label, label_pii_replacements = scrub_pii_like(label)
        split = _normalize_text(str(row.get("split", "test"))) or "test"
        source_row_id = int(row.get("source_row_id", len(rows)))
        rows.append(
            {
                "schema_version": 1,
                "conversation_id": _conversation_id("banking77", split, str(source_row_id), text),
                "source": "PolyAI/banking77",
                "license": POLYAI_LICENSE,
                "created_at": CREATED_AT,
                "messages": [
                    _message("user", text, loss=False),
                    _message("assistant", f"intent:{label}", loss=False),
                ],
                "metadata": {
                    "task": "intent_router_eval",
                    "trainable": False,
                    "source_dataset": POLYAI_DATASET_ID,
                    "source_revision": str(row.get("source_revision", POLYAI_REVISION)),
                    "source_row_id": source_row_id,
                    "source_split": split,
                    "intent": label,
                    "pii_replacements": pii_replacements + label_pii_replacements,
                },
            }
        )
    return rows


def _cluster_bitext_rows(rows: list[dict[str, Any]], threshold: float = 0.1) -> dict[int, int]:
    by_intent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_intent[(str(row["category"]), str(row["intent"]))].append(row)

    cluster_by_source_row: dict[int, int] = {}
    next_cluster = 0
    for group_rows in by_intent.values():
        parent = {int(row["source_row_id"]): int(row["source_row_id"]) for row in group_rows}

        def find(item: int, parent: dict[int, int] = parent) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def union(left: int, right: int, parent: dict[int, int] = parent) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        grams = {
            int(row["source_row_id"]): _token_ngrams(str(row["instruction_key"]))
            for row in group_rows
        }
        for left_index, left in enumerate(group_rows):
            left_id = int(left["source_row_id"])
            for right in group_rows[left_index + 1 :]:
                right_id = int(right["source_row_id"])
                if _jaccard(grams[left_id], grams[right_id]) >= threshold:
                    union(left_id, right_id)
        root_to_cluster: dict[int, int] = {}
        for row in sorted(group_rows, key=lambda item: int(item["source_row_id"])):
            row_id = int(row["source_row_id"])
            root = find(row_id)
            if root not in root_to_cluster:
                root_to_cluster[root] = next_cluster
                next_cluster += 1
            cluster_by_source_row[row_id] = root_to_cluster[root]
    return cluster_by_source_row


def _token_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    tokens = text.split()
    if not tokens:
        return set()
    grams: set[tuple[str, ...]] = set()
    for width in range(1, min(n, len(tokens)) + 1):
        grams.update(
            tuple(tokens[index : index + width])
            for index in range(len(tokens) - width + 1)
        )
    return grams


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _record_to_json(record: PreparedRecord, split: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "conversation_id": record.conversation_id,
        "source": record.source,
        "license": record.license,
        "created_at": CREATED_AT,
        "messages": list(record.messages),
        "metadata": {**record.metadata, "split": split},
    }


def _message(role: str, content: str, loss: bool = False) -> dict[str, Any]:
    return {"role": role, "content": _normalize_text(content), "loss": loss}


def _allowed_use_for_split(split: str) -> list[str]:
    if split == "train":
        return ["tokenizer-training", "model-training"]
    if split == "validation":
        return ["validation", "evaluation"]
    return ["test", "evaluation"]


def _build_lock(
    *,
    bitext_csv: Path,
    bitext_snapshot: Path,
    polyai_snapshot: Path,
    polyai_rows: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "name": "banking-v2-source-lock",
        "created_at": CREATED_AT,
        "summary": {
            "bitext_snapshot": str(bitext_snapshot),
            "polyai_snapshot": str(polyai_snapshot),
            "polyai_rows": polyai_rows,
        },
        "sources": {
            BITEXT_DATASET_ID: {
                "role": "primary-qa-sft",
                "license": BITEXT_LICENSE,
                "revision": BITEXT_REVISION,
                "source_url": _hf_resolve_url(BITEXT_DATASET_ID, BITEXT_REVISION, BITEXT_CSV),
                "download_sha256": file_sha256(bitext_csv),
                "snapshot_path": str(bitext_snapshot),
                "snapshot_sha256": file_sha256(bitext_snapshot),
            },
            POLYAI_DATASET_ID: {
                "role": "intent-router-eval-only",
                "license": POLYAI_LICENSE,
                "revision": POLYAI_REVISION,
                "source_urls": [
                    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/"
                    f"banking_data/{split}.csv"
                    for split in ("train", "test")
                ],
                "snapshot_path": str(polyai_snapshot),
                "snapshot_sha256": file_sha256(polyai_snapshot),
                "trainable": False,
            },
        },
        "quarantine": {
            "talkmap": {
                "trainable": False,
                "reason": "Quarantined metadata only; no source text is acquired or emitted.",
            },
            "rakesh": {
                "trainable": False,
                "reason": "Excluded by user requirement; no source text is acquired or emitted.",
            },
        },
    }


def _load_polyai_rows() -> Iterator[dict[str, Any]]:
    for split in ("train", "test"):
        url = (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/"
            f"banking_data/{split}.csv"
        )
        with urllib.request.urlopen(url, timeout=120) as response:
            text = response.read().decode("utf-8")
        reader = csv.DictReader(text.splitlines())
        for index, row in enumerate(reader):
            yield {
                "source_row_id": index,
                "split": split,
                "text": str(row.get("text", "")),
                "label": str(row.get("label") or row.get("category") or ""),
                "source_dataset": POLYAI_DATASET_ID,
                "source_revision": POLYAI_REVISION,
                "license": POLYAI_LICENSE,
                "trainable": False,
            }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        handle.write(payload)
    tmp_path.replace(path)


def _hf_resolve_url(dataset_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{filename}"


def _read_source_snapshot(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise BankingDataError(f"{path}:{line_number}: invalid JSONL") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_read_source_snapshot(path))


def _write_source_snapshot(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(canonical_json_bytes(value) + b"\n")
    tmp.replace(path)


def _required_text(row: dict[str, Any], field: str) -> str:
    value = _normalize_text(str(row.get(field, "")))
    if not value:
        raise BankingDataError(f"{field} is empty")
    return value


def _normalize_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = "\n".join(WHITESPACE_RE.sub(" ", line).strip() for line in value.split("\n"))
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return value


def _humanize_placeholder(name: str) -> str:
    return NON_WORD_RE.sub(" ", name).strip().lower() or "the requested banking detail"


def _normalized_user_key(text: str) -> str:
    return NON_WORD_RE.sub(" ", _normalize_text(text).casefold()).strip()


def _conversation_id(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return f"bank.v2.{digest[:24]}"


def _ood_bases(split: str) -> list[dict[str, Any]]:
    prompts_by_split = {
        "train": {
            "medical": "What medicine should I take for a fever?",
            "weather": "Will it rain in Chicago tomorrow?",
            "coding": "Write a Python web scraper for me.",
            "travel": "Plan a weekend trip to Paris.",
            "stocks": "Should I buy shares of a specific tech stock?",
            "sports": "Who won the basketball game?",
            "entertainment": "Recommend a science fiction movie.",
            "legal": "Can you draft my divorce paperwork?",
        },
        "validation": {
            "medical": "Can you suggest treatment for a sore throat?",
            "weather": "What will the temperature be in Dallas tonight?",
            "coding": "Debug my JavaScript promise chain.",
            "travel": "Find me a hotel near the beach.",
            "stocks": "Predict tomorrow's market open.",
            "sports": "Explain the offside rule in soccer.",
            "entertainment": "Write lyrics for a pop song.",
            "legal": "Do I have a legal case against my neighbor?",
        },
        "test": {
            "medical": "Should I visit a doctor for this cough?",
            "weather": "Is a snow storm expected in Denver this weekend?",
            "coding": "Create a React component for my blog.",
            "travel": "Choose flights for a vacation to Rome.",
            "stocks": "Tell me which cryptocurrency will rise next week.",
            "sports": "Make a fantasy football lineup for me.",
            "entertainment": "Summarize the plot of a new movie.",
            "legal": "Write a contract for selling my car.",
        },
    }
    return [
        {"topic": topic, "prompts": [prompt]}
        for topic, prompt in prompts_by_split[split].items()
    ]


def _multi_turn_bases(split: str) -> list[dict[str, Any]]:
    terms = {
        "train": {
            "card": "new debit card",
            "activation": "Activate it in the mobile app.",
            "recipient": "my roommate",
            "loan": "auto loan",
            "account": "checking account",
            "ood": "tomorrow's weather",
        },
        "validation": {
            "card": "replacement credit card",
            "activation": "Activate the replacement card online.",
            "recipient": "my landlord",
            "loan": "personal loan",
            "account": "savings account",
            "ood": "a dinner recipe",
        },
        "test": {
            "card": "travel card",
            "activation": "Activate it before my trip.",
            "recipient": "my contractor",
            "loan": "mortgage",
            "account": "student account",
            "ood": "a workout plan",
        },
    }[split]
    return [
        {
            "base_issue": f"{split}_card_activation_clarification",
            "category": "CARD",
            "intent": "activate_card",
            "turn_pattern": "clarification",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT, "loss": False},
                {
                    "role": "user",
                    "content": f"I need help with my {terms['card']}.",
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": (
                        f"I can help with the {terms['card']}. Are you trying to "
                        "activate it, replace it, or report a problem?"
                    ),
                    "loss": True,
                },
                {"role": "user", "content": terms["activation"], "loss": False},
                {
                    "role": "assistant",
                    "content": (
                        f"Open the mobile banking app for the {terms['card']}, go to "
                        "card services, choose activate card, and follow the identity "
                        "checks shown there."
                    ),
                    "loss": True,
                },
            ],
        },
        {
            "base_issue": f"{split}_transfer_follow_up",
            "category": "TRANSFER",
            "intent": "make_transfer",
            "turn_pattern": "follow_up",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT, "loss": False},
                {
                    "role": "user",
                    "content": f"How do I send money to {terms['recipient']}?",
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": (
                        f"Use transfers, add or select {terms['recipient']}, enter the "
                        "amount, review the details, and confirm only if everything is "
                        "correct."
                    ),
                    "loss": True,
                },
                {
                    "role": "user",
                    "content": f"What if I typed the wrong amount for {terms['recipient']}?",
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": (
                        "Do not confirm it. Go back, correct the amount, and review "
                        f"the payment details for {terms['recipient']} again before "
                        "submitting."
                    ),
                    "loss": True,
                },
            ],
        },
        {
            "base_issue": f"{split}_loan_correction",
            "category": "LOAN",
            "intent": "check_loan_payments",
            "turn_pattern": "correction",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT, "loss": False},
                {
                    "role": "user",
                    "content": f"I want to close my {terms['loan']}.",
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": (
                        f"I can explain payoff or cancellation options for the {terms['loan']}. "
                        "Are you asking about early payoff or checking payment details?"
                    ),
                    "loss": True,
                },
                {
                    "role": "user",
                    "content": f"Sorry, I only need to check upcoming {terms['loan']} payments.",
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": (
                        "Sign in to online or mobile banking and open the "
                        f"{terms['loan']}. The payment schedule should show upcoming "
                        "due dates and amounts."
                    ),
                    "loss": True,
                },
            ],
        },
        {
            "base_issue": f"{split}_account_to_ood",
            "category": "ACCOUNT",
            "intent": "create_account",
            "turn_pattern": "in_domain_to_ood",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT, "loss": False},
                {
                    "role": "user",
                    "content": f"How do I open a {terms['account']}?",
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": (
                        f"You can usually open a {terms['account']} from the bank website "
                        "or a branch. Review fees, eligibility, required identification, "
                        "and funding options before applying."
                    ),
                    "loss": True,
                },
                {"role": "user", "content": f"Also tell me {terms['ood']}.", "loss": False},
                {"role": "assistant", "content": BANKING_V2_CANNED_OOD_RESPONSE, "loss": True},
            ],
        },
    ]


def _render_data_card(report: dict[str, Any]) -> str:
    return (
        "# Banking-v2 data card\n\n"
        "This is a retail-banking SFT corpus for a separate banking-v2 adaptation track. "
        "It is not evidence that the corpus is sufficient for 9B pretraining.\n\n"
        f"- Total conversations: {report['summary']['total_conversations']}\n"
        f"- Corpus fingerprint: `{report['summary']['corpus_fingerprint']}`\n"
        f"- Placeholder check: {report['checks']['unresolved_placeholders']} unresolved\n"
        f"- Cross-split normalized user duplicates: "
        f"{report['checks']['cross_split_normalized_user_duplicates']}\n"
        f"- OOD response: `{BANKING_V2_CANNED_OOD_RESPONSE}`\n\n"
        "Known integration gaps are listed in `preparation-report.json`."
    )


if __name__ == "__main__":
    raise SystemExit(main())
