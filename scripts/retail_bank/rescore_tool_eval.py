#!/usr/bin/env python
"""Rescore persisted model predictions against corrected, prompt-equivalent metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hello_slm.banking_tool_eval import (  # noqa: E402
    StaticPredictionModel,
    TaggedJsonToolAdapter,
    evaluate_records,
    load_predictions_jsonl,
    release_gate_failures,
)
from hello_slm.config import canonical_json_bytes, file_sha256  # noqa: E402

REVISION_HEX_LENGTH = 40


class RescoreError(ValueError):
    """Raised when persisted predictions cannot be safely rescored."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-dataset-repo",
        default="spkc83/retail-bank-servicing-alignment-sft",
    )
    parser.add_argument("--generation-dataset-revision", required=True)
    parser.add_argument(
        "--scoring-dataset-repo",
        default="spkc83/retail-bank-servicing-alignment-sft",
    )
    parser.add_argument("--scoring-dataset-revision", required=True)
    parser.add_argument(
        "--prediction-repo",
        default="spkc83/retail-bank-servicing-agent-9b",
    )
    parser.add_argument("--prediction-revision", required=True)
    parser.add_argument("--prediction-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--template-hash", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/retail-bank-tool-eval-rescore"),
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--enforce-release-gates", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args(argv)


def validate_exact_revision(value: str, *, field: str) -> None:
    if len(value) != REVISION_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RescoreError(f"{field} must be an exact 40-character lowercase Git revision")


def verify_prompt_equivalence(
    generation_records: Sequence[Mapping[str, Any]],
    scoring_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    generation_ids = [str(record.get("record_id")) for record in generation_records]
    scoring_ids = [str(record.get("record_id")) for record in scoring_records]
    if generation_ids != scoring_ids:
        raise RescoreError("generation and scoring record IDs or ordering differ")
    messages = []
    changed_expected = 0
    for generation, scoring, record_id in zip(
        generation_records,
        scoring_records,
        generation_ids,
        strict=True,
    ):
        if generation.get("messages") != scoring.get("messages"):
            raise RescoreError(f"generation and scoring messages differ for record {record_id}")
        messages.append(generation.get("messages"))
        changed_expected += generation.get("expected") != scoring.get("expected")
    return {
        "record_count": len(generation_records),
        "messages_sha256": "sha256:" + hashlib.sha256(canonical_json_bytes(messages)).hexdigest(),
        "changed_expected_records": changed_expected,
    }


def load_exact_predictions(
    path: Path,
    generation_records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    outputs = load_predictions_jsonl(path)
    expected_ids = {str(record.get("record_id")) for record in generation_records}
    actual_ids = set(outputs)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise RescoreError(
            f"prediction coverage mismatch: missing={missing[:5]!r}, extra={extra[:5]!r}"
        )
    return outputs


def load_manifest_records(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("tool_sft")
    if not isinstance(entries, list):
        raise RescoreError(f"manifest {manifest_path} does not declare tool_sft splits")
    paths = [
        manifest_path.parent / str(entry["path"])
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("name") == split
        and entry.get("included", True)
    ]
    records: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise RescoreError(f"manifest {manifest_path} has no records for split {split!r}")
    return records


def snapshot_manifest(
    repo_id: str,
    revision: str,
    *,
    token: str | None,
) -> Path:
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            token=token,
            allow_patterns=["manifest.json", "*.jsonl"],
        )
    )
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise RescoreError(f"dataset manifest is unavailable at revision {revision}")
    return manifest


def snapshot_prediction(
    repo_id: str,
    revision: str,
    path_in_repo: str,
    *,
    token: str | None,
) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            filename=path_in_repo,
            token=token,
        )
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    revision_fields = (
        (args.generation_dataset_revision, "--generation-dataset-revision"),
        (args.scoring_dataset_revision, "--scoring-dataset-revision"),
        (args.prediction_revision, "--prediction-revision"),
        (args.model_revision, "--model-revision"),
    )
    for revision, field in revision_fields:
        validate_exact_revision(str(revision), field=field)

    generation_manifest = snapshot_manifest(
        args.generation_dataset_repo,
        args.generation_dataset_revision,
        token=args.token,
    )
    scoring_manifest = snapshot_manifest(
        args.scoring_dataset_repo,
        args.scoring_dataset_revision,
        token=args.token,
    )
    generation_records = load_manifest_records(generation_manifest, args.split)
    scoring_records = load_manifest_records(scoring_manifest, args.split)
    equivalence = verify_prompt_equivalence(generation_records, scoring_records)
    prediction_path = snapshot_prediction(
        args.prediction_repo,
        args.prediction_revision,
        args.prediction_path,
        token=args.token,
    )
    outputs = load_exact_predictions(prediction_path, generation_records)
    report = evaluate_records(
        scoring_records,
        model=StaticPredictionModel(outputs),
        adapter=TaggedJsonToolAdapter(template_hash=args.template_hash),
        checkpoint_revision=args.model_revision,
    )
    failures = release_gate_failures(report)
    slug = f"{args.model_revision[:12]}-{args.scoring_dataset_revision[:12]}-{args.split}"
    report_path = args.output_dir / f"report-{slug}.json"
    metadata_path = args.output_dir / f"metadata-{slug}.json"
    write_json(report_path, report)
    metadata: dict[str, Any] = {
        "contract": "banking-tool-eval-rescore-metadata/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "model": {
            "repo": args.prediction_repo,
            "weights_revision": args.model_revision,
        },
        "generation_dataset": {
            "repo": args.generation_dataset_repo,
            "revision": args.generation_dataset_revision,
            "manifest_sha256": file_sha256(generation_manifest),
        },
        "scoring_dataset": {
            "repo": args.scoring_dataset_repo,
            "revision": args.scoring_dataset_revision,
            "manifest_sha256": file_sha256(scoring_manifest),
        },
        "persisted_predictions": {
            "repo": args.prediction_repo,
            "artifact_revision": args.prediction_revision,
            "path": args.prediction_path,
            "sha256": file_sha256(prediction_path),
        },
        "prompt_equivalence": equivalence,
        "adapter_template_hash": args.template_hash,
        "report": {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
        },
        "release_gate": {
            "eligible": not failures,
            "enforced": bool(args.enforce_release_gates),
            "failures": failures,
        },
    }
    write_json(metadata_path, metadata)
    if args.push_to_hub:
        from huggingface_hub import HfApi

        path_in_repo = (
            f"evaluation/{args.model_revision[:12]}-"
            f"{args.scoring_dataset_revision[:12]}-rescore"
        )
        commit = HfApi(token=args.token).upload_folder(
            repo_id=args.prediction_repo,
            repo_type="model",
            folder_path=args.output_dir,
            path_in_repo=path_in_repo,
            commit_message="Publish prompt-equivalent frozen evaluation rescore",
        )
        metadata["publication"] = {
            "path_in_repo": path_in_repo,
            "revision": commit.oid,
        }
        write_json(metadata_path, metadata)
    if args.enforce_release_gates and failures:
        raise RescoreError("frozen evaluation release gates failed: " + "; ".join(failures))
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    metadata = run(parse_args(argv))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
