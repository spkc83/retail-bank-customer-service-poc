#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.12.0",
#   "huggingface-hub==1.22.0",
#   "peft==0.18.1",
#   "safetensors==0.8.0",
#   "torch>=2.9,<3",
#   "transformers==5.13.0",
# ]
# ///
"""Recover a continuation release by exporting persisted adapter candidates."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hf_job_finalize_tool_sft import (  # type: ignore[import-not-found]
    ADAPTER_ALLOWLIST,
    MERGED_ALLOWLIST,
    validate_parity,
)

MODEL_REPO = "spkc83/retail-bank-agent-9b"
BASE_MODEL = "ibm-granite/granite-4.1-8b"
BASE_REVISION = "1504002f650e656a0a3789d99574df12e3e94ed0"
DEFAULT_OUTPUT_ROOT = Path(
    "/data/retail-bank-agent-9b-continuation-68e96a7d-00c4ba1b"
)
MINIMUM_ARGMAX_AGREEMENT = 0.999
MAXIMUM_LOGIT_DIFFERENCE = 0.3
MAXIMUM_P999_DIFFERENCE = 0.07


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--recovery-source-commit", required=True)
    parser.add_argument("--training-source-commit", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--parent-model-revision", required=True)
    parser.add_argument("--training-job", required=True)
    parser.add_argument("--selected-adapter-subdir", default="adapter")
    parser.add_argument("--selected-step", type=int, default=600)
    return parser.parse_args(argv)


def require_exact_revision(value: str, *, field: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be an exact 40-character lowercase Git revision")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_argument(command: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"training job command does not contain exactly one {flag}")
    return str(command[positions[0] + 1])


def validate_training_provenance(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    training_job: Any,
    artifact_mtimes: Mapping[str, float],
) -> dict[str, Any]:
    fingerprint = metadata.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise RuntimeError("continuation metadata has no fingerprint")
    source_model = fingerprint.get("source_model")
    dataset_identity = fingerprint.get("dataset_identity")
    if not isinstance(source_model, dict) or not isinstance(dataset_identity, dict):
        raise RuntimeError("continuation metadata provenance is incomplete")
    expected = {
        "parent_model_revision": args.parent_model_revision,
        "dataset_revision": args.dataset_revision,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
    }
    observed = {
        "parent_model_revision": source_model.get("revision"),
        "dataset_revision": dataset_identity.get("revision"),
        "base_model": fingerprint.get("base_model"),
        "base_revision": fingerprint.get("base_revision"),
    }
    mismatches = [
        f"{name}: metadata={observed[name]!r}, requested={value!r}"
        for name, value in expected.items()
        if observed[name] != value
    ]
    command = list(getattr(training_job, "command", ()))
    command_expected = {
        "--source-commit": args.training_source_commit,
        "--dataset-revision": args.dataset_revision,
        "--source-model-revision": args.parent_model_revision,
        "--output-dir": str(args.output_root),
        "--max-steps": "600",
    }
    for flag, value in command_expected.items():
        observed_value = command_argument(command, flag)
        if observed_value != value:
            mismatches.append(
                f"{flag}: job={observed_value!r}, requested={value!r}"
            )
    if str(getattr(training_job, "id", "")) != args.training_job:
        mismatches.append("inspected training job ID does not match --training-job")
    started_at = getattr(training_job, "started_at", None)
    finished_at = getattr(training_job, "finished_at", None)
    if started_at is None or finished_at is None:
        mismatches.append("training job has no complete execution time window")
    else:
        started_timestamp = float(started_at.timestamp())
        finished_timestamp = float(finished_at.timestamp())
        for name, modified_at in artifact_mtimes.items():
            if not started_timestamp <= modified_at <= finished_timestamp + 1.0:
                mismatches.append(
                    f"{name} mtime {modified_at} is outside training job window "
                    f"{started_timestamp}..{finished_timestamp}"
                )
        metadata_created_at = float(metadata.get("created_at_unix", 0))
        if not started_timestamp <= metadata_created_at <= finished_timestamp + 1.0:
            mismatches.append(
                "continuation metadata creation time is outside training job window"
            )
    if mismatches:
        raise RuntimeError(
            "persisted continuation provenance mismatch: " + "; ".join(mismatches)
        )
    return {
        "training_job_id": args.training_job,
        "training_job_command_verified": True,
        "training_job_artifact_window_verified": True,
        "training_source_commit": args.training_source_commit,
        "dataset_revision": str(dataset_identity["revision"]),
        "dataset_manifest_sha256": str(dataset_identity["manifest_sha256"]),
        "parent_model_revision": str(source_model["revision"]),
        "base_model": str(fingerprint["base_model"]),
        "base_revision": str(fingerprint["base_revision"]),
        "output_root": str(args.output_root),
        "max_steps": 600,
    }


def run_candidate(
    args: argparse.Namespace,
    *,
    merged_subdir: str,
    merge_dtype: str,
    release_dtype: str,
    inference_dtype: str,
    report_name: str,
    adapter_subdir: str,
    selected_step: int,
) -> dict[str, Any]:
    remerge_script = SCRIPT_DIR / "hf_job_remerge_tool_sft.py"
    parity_script = SCRIPT_DIR / "hf_job_merge_parity.py"
    subprocess.run(
        [
            sys.executable,
            str(remerge_script),
            "--output-root",
            str(args.output_root),
            "--output-subdir",
            merged_subdir,
            "--adapter-subdir",
            adapter_subdir,
            "--base-model",
            args.base_model,
            "--base-revision",
            args.base_revision,
            "--merge-dtype",
            merge_dtype,
            "--release-dtype",
            release_dtype,
            "--report-name",
            report_name,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(parity_script),
            "--output-root",
            str(args.output_root),
            "--merged-subdir",
            merged_subdir,
            "--adapter-subdir",
            adapter_subdir,
            "--base-model",
            args.base_model,
            "--base-revision",
            args.base_revision,
            "--inference-dtype",
            inference_dtype,
        ],
        check=True,
    )
    parity_name = (
        f"merge_parity_diagnostics_{merged_subdir}_{inference_dtype}.json"
    )
    parity = read_json(args.output_root / parity_name)
    try:
        metrics = validate_parity(
            parity,
            minimum_argmax_agreement=MINIMUM_ARGMAX_AGREEMENT,
            maximum_logit_difference=MAXIMUM_LOGIT_DIFFERENCE,
            maximum_p999_difference=MAXIMUM_P999_DIFFERENCE,
        )
    except RuntimeError as error:
        return {
            "passed": False,
            "merged_subdir": merged_subdir,
            "merge_dtype": merge_dtype,
            "release_dtype": release_dtype,
            "inference_dtype": inference_dtype,
            "parity_report": parity_name,
            "adapter_subdir": adapter_subdir,
            "selected_step": selected_step,
            "metrics": parity.get("metrics"),
            "failure": str(error),
        }
    return {
        "passed": True,
        "merged_subdir": merged_subdir,
        "merge_dtype": merge_dtype,
        "release_dtype": release_dtype,
        "inference_dtype": inference_dtype,
        "remerge_report": report_name,
        "parity_report": parity_name,
        "adapter_subdir": adapter_subdir,
        "selected_step": selected_step,
        "metrics": metrics,
    }


def publish(
    args: argparse.Namespace,
    *,
    candidate: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata_path = args.output_root / "continuation_training_metadata.json"
    metadata = read_json(metadata_path)
    if metadata.get("step") != 600:
        raise RuntimeError("continuation metadata does not represent step 600")
    merged_dir = args.output_root / str(candidate["merged_subdir"])
    adapter_dir = args.output_root / str(candidate["adapter_subdir"])
    trainer_state_path = adapter_dir / "trainer_state.json"
    api = HfApi(token=os.environ["HF_TOKEN"])
    namespace = args.model_repo.split("/", 1)[0]
    training_job = api.inspect_job(
        job_id=args.training_job,
        namespace=namespace,
    )
    verified_provenance = validate_training_provenance(
        args,
        metadata,
        training_job,
        {
            "continuation_training_metadata": metadata_path.stat().st_mtime,
            "selected_adapter": (
                adapter_dir / "adapter_model.safetensors"
            ).stat().st_mtime,
            "selected_trainer_state": trainer_state_path.stat().st_mtime,
        },
    )
    trainer_state = read_json(trainer_state_path)
    selected_step = int(candidate["selected_step"])
    if int(trainer_state.get("global_step", 0)) != selected_step:
        raise RuntimeError("selected checkpoint trainer state does not match its step")
    selected_eval = [
        entry
        for entry in trainer_state.get("log_history", ())
        if isinstance(entry, dict)
        and "eval_loss" in entry
        and int(entry.get("step", 0)) <= selected_step
    ]
    result = {
        "contract": "banking-v3-continuation-sft-result/v1",
        "worker": "cloud_recover_continuation_export",
        "run_completed_steps": 600,
        "selected_step": selected_step,
        "selected_adapter_subdir": str(candidate["adapter_subdir"]),
        "recovery_source_commit": args.recovery_source_commit,
        "verified_training_provenance": verified_provenance,
        "recovery_job": os.environ.get("HF_JOB_ID"),
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "run_final_train_metrics": metadata.get("train_metrics"),
        "run_final_eval_metrics": metadata.get("eval_metrics"),
        "selected_checkpoint_last_eval": selected_eval[-1] if selected_eval else None,
        "selected_checkpoint_trainer_state_sha256": sha256(trainer_state_path),
        "release": {
            "selected_candidate": candidate,
            "attempts": attempts,
            "acceptance": {
                "minimum_argmax_agreement": MINIMUM_ARGMAX_AGREEMENT,
                "maximum_logit_difference": MAXIMUM_LOGIT_DIFFERENCE,
                "maximum_p999_difference": MAXIMUM_P999_DIFFERENCE,
                "all_greedy_generations_equal": True,
                "all_logit_differences_finite": True,
            },
        },
        "adapter_sha256": sha256(adapter_dir / "adapter_model.safetensors"),
        "merged_model_sha256": sha256(merged_dir / "model.safetensors"),
        "pushed_to_hub": args.model_repo,
    }
    result_path = args.output_root / "continuation_training_result.json"
    write_json(result_path, result)
    evidence = (
        (metadata_path, "training_metadata.json", "Record continuation SFT metadata"),
        (
            args.output_root / str(candidate["parity_report"]),
            "merge_parity_diagnostics.json",
            "Record continuation merge parity diagnostics",
        ),
        (
            args.output_root / str(candidate["remerge_report"]),
            "fp16_remerge.json",
            "Record continuation merge provenance",
        ),
        (result_path, "training_result.json", "Record continuation SFT result"),
        (
            trainer_state_path,
            "selected_checkpoint/trainer_state.json",
            "Record selected checkpoint trainer state",
        ),
    )
    del candidate, attempts
    operations = [
        CommitOperationAdd(
            path_in_repo=name,
            path_or_fileobj=merged_dir / name,
        )
        for name in MERGED_ALLOWLIST
    ]
    operations.extend(
        CommitOperationAdd(
            path_in_repo=f"adapter/{name}",
            path_or_fileobj=adapter_dir / name,
        )
        for name in ADAPTER_ALLOWLIST
    )
    operations.extend(
        CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=local_path)
        for local_path, remote_path, _message in evidence
    )
    api.create_repo(args.model_repo, repo_type="model", private=False, exist_ok=True)
    release_commit = api.create_commit(
        repo_id=args.model_repo,
        repo_type="model",
        operations=operations,
        commit_message="Publish continuation SFT release bundle",
    )
    weights_revision = str(release_commit.oid)
    require_exact_revision(weights_revision, field="weights revision")
    result["weights_revision"] = weights_revision
    write_json(result_path, result)
    provenance_commit = api.create_commit(
        repo_id=args.model_repo,
        repo_type="model",
        operations=[
            CommitOperationAdd(
                path_in_repo="training_result.json",
                path_or_fileobj=result_path,
            )
        ],
        commit_message="Record exact continuation release revision",
    )
    result["provenance_revision"] = str(provenance_commit.oid)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for field in (
        "recovery_source_commit",
        "training_source_commit",
        "dataset_revision",
        "parent_model_revision",
    ):
        require_exact_revision(getattr(args, field), field=f"--{field.replace('_', '-')}")
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN must be passed as a Hugging Face Job secret")
    selected_adapter = args.output_root / args.selected_adapter_subdir
    if not (selected_adapter / "adapter_model.safetensors").is_file():
        raise RuntimeError("persisted continuation adapter is unavailable")
    if args.selected_step < 1:
        raise ValueError("--selected-step must be positive")

    attempts: list[dict[str, Any]] = []
    step_label = f"step{args.selected_step}"
    candidates = (
        {
            "merged_subdir": f"merged-{step_label}-fp16-native",
            "merge_dtype": "float16",
            "release_dtype": "float16",
            "inference_dtype": "float16",
            "report_name": f"{step_label}_fp16_native_remerge.json",
            "adapter_subdir": args.selected_adapter_subdir,
            "selected_step": args.selected_step,
        },
        {
            "merged_subdir": f"merged-{step_label}-fp32-fp16",
            "merge_dtype": "float32",
            "release_dtype": "float16",
            "inference_dtype": "float16",
            "report_name": f"{step_label}_fp32_fp16_remerge.json",
            "adapter_subdir": args.selected_adapter_subdir,
            "selected_step": args.selected_step,
        },
    )
    for candidate_spec in candidates:
        candidate = run_candidate(args, **candidate_spec)
        attempts.append(candidate)
        if candidate["passed"]:
            result = publish(args, candidate=candidate, attempts=attempts)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
    report_path = args.output_root / "continuation_export_recovery.json"
    write_json(report_path, {"passed": False, "attempts": attempts})
    print(report_path.read_text(encoding="utf-8"))
    raise RuntimeError("no export candidate passed the unchanged parity gates")


if __name__ == "__main__":
    raise SystemExit(main())
