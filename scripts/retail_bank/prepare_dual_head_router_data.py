#!/usr/bin/env python
"""Prepare governed Banking77 + CLINC150 data for the dual-head router."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tempfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hello_slm.banking_router_data import (
    CLINC_CONVERSATIONAL_IN_DOMAIN_LABELS,
    CLINC_SUPPORTED_BANKING_LABELS,
    build_router_splits,
)

CLINC_URL = "https://archive.ics.uci.edu/static/public/570/clinc150.zip"
CLINC_ZIP_SHA256 = "0d8ecc3e1edd7b25cabde0177544ce536ddf773844bc80ef1a75f36e7f030ea2"
CLINC_MEMBER = "clinc150_uci/data_oos_plus.json"
CLINC_MEMBER_SHA256 = "bfcca9ae515623541dc1983c94c4ed7cae9d26b42ae47d74b972e51bb6f7a21f"
BANKING77_DATASET_ID = "PolyAI/banking77"
BANKING77_RELEASE_REVISION = "90d4e2ee5521c04fc1488f065b8b083658768c57"
BANKING77_SOURCE_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BANKING77_URLS = {
    split: (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
        f"{BANKING77_SOURCE_REVISION}/banking_data/{split}.csv"
    )
    for split in ("train", "test")
}
BANKING77_SHA256 = {
    "train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
BANKING77_SNAPSHOT_SHA256 = (
    "22ce056724069f431b477aa8478f1a42ce31286ad595cb7e53a838173052b340"
)
DEFAULT_OUTPUT_DIR = Path("data/banking-router-v1")
DEFAULT_RELEASE_LOCK = Path("data/sources/banking-router-v1.lock.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-lock",
        type=Path,
        help="Output lock path; defaults to SOURCE_LOCK.json inside --output-dir.",
    )
    parser.add_argument(
        "--expected-release-lock",
        type=Path,
        default=DEFAULT_RELEASE_LOCK,
        help="Tracked release lock whose split digests must be reproduced.",
    )
    parser.add_argument(
        "--skip-release-digest-check",
        action="store_true",
        help="Allow intentional experimental splits that differ from the released dataset.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7101)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    banking_payloads = {
        split: _download(url) for split, url in BANKING77_URLS.items()
    }
    for split, payload in banking_payloads.items():
        actual_sha = _bytes_sha256(payload)
        expected_sha = BANKING77_SHA256[split]
        if actual_sha != expected_sha:
            raise ValueError(
                f"Banking77 {split} CSV digest mismatch: "
                f"expected {expected_sha}, got {actual_sha}"
            )
    banking_rows = parse_banking77_csvs(banking_payloads)
    actual_banking_sha = _bytes_sha256(_jsonl_bytes(banking_rows))
    if actual_banking_sha != BANKING77_SNAPSHOT_SHA256:
        raise ValueError(
            "Banking77 normalized snapshot digest mismatch: "
            f"expected {BANKING77_SNAPSHOT_SHA256}, got {actual_banking_sha}"
        )

    clinc_zip = _download(CLINC_URL)
    if _bytes_sha256(clinc_zip) != CLINC_ZIP_SHA256:
        raise ValueError("CLINC150 archive digest mismatch")
    with zipfile.ZipFile(io.BytesIO(clinc_zip)) as archive:
        clinc_bytes = archive.read(CLINC_MEMBER)
    if _bytes_sha256(clinc_bytes) != CLINC_MEMBER_SHA256:
        raise ValueError("CLINC150 data_oos_plus.json digest mismatch")
    clinc_payload = json.loads(clinc_bytes)

    splits, report = build_router_splits(
        banking_rows,
        clinc_payload,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if report["pii_matches"] != 0:
        raise ValueError(f"router data contains {report['pii_matches']} PII-like matches")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_entries = []
    for split, rows in splits.items():
        path = args.output_dir / f"{split}.jsonl"
        _write_jsonl(path, rows)
        split_entries.append(
            {
                "name": split,
                "path": path.name,
                "rows": len(rows),
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
                "allowed_use": (
                    ["dual-head-router-training"]
                    if split == "train"
                    else ["dual-head-router-calibration", "dual-head-router-evaluation"]
                ),
            }
        )

    created_at = datetime.now(UTC).isoformat()
    manifest = {
        "contract": "banking-dual-head-router-data",
        "format_version": 1,
        "created_at": created_at,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "sources": {
            "PolyAI/banking77": {
                "revision": BANKING77_RELEASE_REVISION,
                "source_repository_revision": BANKING77_SOURCE_REVISION,
                "source_urls": BANKING77_URLS,
                "source_sha256": BANKING77_SHA256,
                "snapshot_sha256": actual_banking_sha,
                "license": "CC-BY-4.0",
                "allowed_use": ["dual-head-router-training", "evaluation"],
                "generative_sft": False,
            },
            "UCI/clinc150": {
                "doi": "10.24432/C5MP58",
                "url": CLINC_URL,
                "archive_sha256": CLINC_ZIP_SHA256,
                "member": CLINC_MEMBER,
                "member_sha256": CLINC_MEMBER_SHA256,
                "license": "CC-BY-4.0",
                "allowed_use": ["dual-head-domain-training", "evaluation"],
            },
        },
        "clinc_supported_banking_labels": sorted(CLINC_SUPPORTED_BANKING_LABELS),
        "clinc_conversational_in_domain_labels": sorted(
            CLINC_CONVERSATIONAL_IN_DOMAIN_LABELS
        ),
        "splits": split_entries,
        "report": report,
        "review_status": "automated-policy-pass",
        "signed": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_data_card(args.output_dir / "README.md", manifest)

    if not args.skip_release_digest_check:
        verify_release_split_digests(
            split_entries,
            _read_json(args.expected_release_lock),
        )

    source_lock = {
        "contract": "banking-dual-head-router-source-lock",
        "format_version": 1,
        "created_at": created_at,
        "sources": manifest["sources"],
        "prepared_manifest_sha256": _file_sha256(manifest_path),
        "prepared_split_sha256": {
            entry["name"]: entry["sha256"] for entry in split_entries
        },
    }
    source_lock_path = args.source_lock or (args.output_dir / "SOURCE_LOCK.json")
    source_lock_path.parent.mkdir(parents=True, exist_ok=True)
    source_lock_path.write_text(
        json.dumps(source_lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _download(url: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="retail-bank-router-"):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "retail-bank-servicing/0.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()


def parse_banking77_csvs(payloads: dict[str, bytes]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        payload = payloads.get(split)
        if payload is None:
            raise ValueError(f"Banking77 payload is missing the {split} split")
        # Keep the released snapshot normalization: the original audited
        # Banking77 importer passed split lines to DictReader, which removes
        # decorative newlines inside a small number of quoted cells.
        reader = csv.DictReader(payload.decode("utf-8-sig").splitlines())
        for index, row in enumerate(reader):
            text = str(row.get("text", ""))
            label = str(row.get("label") or row.get("category") or "")
            if not text or not label:
                raise ValueError(f"Banking77 {split} row {index} is incomplete")
            rows.append(
                {
                    "source_row_id": index,
                    "split": split,
                    "text": text,
                    "label": label,
                    "source_dataset": BANKING77_DATASET_ID,
                    "source_revision": BANKING77_RELEASE_REVISION,
                    "license": "CC-BY-4.0",
                    "trainable": False,
                }
            )
    return rows


def verify_release_split_digests(
    split_entries: list[dict[str, Any]],
    release_lock: dict[str, Any],
) -> None:
    expected = release_lock.get("prepared_split_sha256")
    if not isinstance(expected, dict):
        raise ValueError("release lock is missing prepared_split_sha256")
    actual = {str(entry["name"]): str(entry["sha256"]) for entry in split_entries}
    for split in ("train", "validation", "test"):
        if actual.get(split) != expected.get(split):
            raise ValueError(
                f"{split} split digest drift: expected {expected.get(split)}, "
                f"got {actual.get(split)}"
            )


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True) + "\n"
        for row in rows
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_data_card(path: Path, manifest: dict[str, Any]) -> None:
    counts = manifest["report"]["split_counts"]
    path.write_text(
        "\n".join(
            [
                "---",
                "license: cc-by-4.0",
                "task_categories:",
                "  - text-classification",
                "language:",
                "  - en",
                "---",
                "",
                "# Retail Bank dual-head router data",
                "",
                "Governed classifier-only data derived from PolyAI Banking77 and UCI CLINC150.",
                "It is not included in generative SFT.",
                "",
                f"- Train rows: {counts['train']}",
                f"- Validation rows: {counts['validation']}",
                f"- Test rows: {counts['test']}",
                "- Domain labels: OOD=0, supported retail banking=1",
                (
                    "- Supported domain includes greetings, thanks, goodbyes, and "
                    "assistant-identity questions"
                ),
                "- Intent labels: 77 Banking77 intents; `-100` means no intent supervision",
                "- Licenses: CC-BY-4.0",
                "",
                (
                    "See `manifest.json` for source revisions, hashes, mapping policy, "
                    "and audit counts."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
