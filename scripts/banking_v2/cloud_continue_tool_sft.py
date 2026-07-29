#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.12.0",
#   "datasets==4.5.0",
#   "huggingface-hub==1.22.0",
#   "peft==0.18.1",
#   "safetensors==0.8.0",
#   "torch>=2.9,<3",
#   "trackio>=0.33,<0.34",
#   "transformers==5.13.0",
#   "trl==0.26.2",
# ]
# ///
"""Continuation SFT from the retained banking-v3 LoRA adapter.

This worker does not rerun the original 3000-step SFT. It loads the exact
Granite base plus a retained adapter from a pinned model revision, continues
LoRA training on an oversampled sequential/clarification mix, then delegates
release to the existing FP32-accumulated FP16 merge and parity gates.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cloud_train_tool_sft import (  # type: ignore[import-not-found]
    PUBLIC_BANKING_TOOL_MANIFEST,
    TRAINING_SEED,
    collate_pretokenized,
    load_manifest_records,
    seed_training,
    sha256_file,
    tf32_supported,
    tokenize_records,
)
from hf_job_finalize_tool_sft import (  # type: ignore[import-not-found]
    ADAPTER_ALLOWLIST,
    MERGED_ALLOWLIST,
    validate_parity,
)
from hello_slm.banking_tool_wire import ToolWireAdapter

REMOTE_CONFIRMATION_ENV = "RETAIL_BANK_ALLOW_REMOTE_CONTINUATION_SFT"
REMOTE_CONFIRMATION_VALUE = "banking-v3-continuation-sft"
DATASET_REPO = "spkc83/retail-bank-agent-sft"
MODEL_REPO = "spkc83/retail-bank-agent-9b"
BASE_MODEL = "ibm-granite/granite-4.1-8b"
BASE_REVISION = "1504002f650e656a0a3789d99574df12e3e94ed0"
DEFAULT_SOURCE_MODEL_REVISION = "00c4ba1be926fc26dbc1f5311a4fd037462be1c1"
DEFAULT_MANIFEST = "data/banking-v3-tool-sft/manifest.json"
DEFAULT_OUTPUT_DIR = "/data/retail-bank-agent-9b-continuation"

@dataclass(frozen=True)
class ContinuationConfig:
    manifest: Path
    output_dir: Path
    source_model_repo: str
    source_model_revision: str
    base_model: str
    base_revision: str
    family: str
    hub_dest: str
    max_steps: int
    max_train_seconds: int
    batch_size: int
    gradient_accumulation_steps: int
    max_seq_len: int
    learning_rate: float
    checkpoint_every: int
    sequential_multiplier: int
    clarification_multiplier: int
    dry_run: bool
    allow_remote_execution: bool
    push_to_hub: bool
    trackio_project: str | None
    trackio_run_name: str | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-model-repo", default=MODEL_REPO)
    parser.add_argument("--source-model-revision", default=DEFAULT_SOURCE_MODEL_REVISION)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--family", default="granite")
    parser.add_argument("--hub-dest", default=MODEL_REPO)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-train-seconds", type=int, default=9_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--sequential-multiplier", type=int, default=5)
    parser.add_argument("--clarification-multiplier", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute-remote", action="store_false", dest="dry_run")
    parser.add_argument("--allow-remote-execution", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--trackio-project")
    parser.add_argument("--trackio-run-name")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ContinuationConfig:
    return ContinuationConfig(
        manifest=Path(args.manifest),
        output_dir=Path(args.output_dir),
        source_model_repo=str(args.source_model_repo),
        source_model_revision=str(args.source_model_revision),
        base_model=str(args.base_model),
        base_revision=str(args.base_revision),
        family=str(args.family),
        hub_dest=str(args.hub_dest),
        max_steps=int(args.max_steps),
        max_train_seconds=int(args.max_train_seconds),
        batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        max_seq_len=int(args.max_seq_len),
        learning_rate=float(args.learning_rate),
        checkpoint_every=int(args.checkpoint_every),
        sequential_multiplier=int(args.sequential_multiplier),
        clarification_multiplier=int(args.clarification_multiplier),
        dry_run=bool(args.dry_run),
        allow_remote_execution=bool(args.allow_remote_execution),
        push_to_hub=bool(args.push_to_hub),
        trackio_project=args.trackio_project,
        trackio_run_name=args.trackio_run_name,
    )


def require_exact_revision(value: str, *, field: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be an exact 40-character lowercase Git revision")


def remote_execution_allowed(config: ContinuationConfig) -> bool:
    return bool(
        not config.dry_run
        and config.allow_remote_execution
        and os.environ.get(REMOTE_CONFIRMATION_ENV) == REMOTE_CONFIRMATION_VALUE
    )


def assert_remote_execution_allowed(config: ContinuationConfig) -> None:
    if not remote_execution_allowed(config):
        raise PermissionError(
            "Continuation SFT requires --execute-remote, --allow-remote-execution, "
            f"and {REMOTE_CONFIRMATION_ENV}={REMOTE_CONFIRMATION_VALUE}."
        )


def assistant_tool_call_count(record: Mapping[str, Any]) -> int:
    return sum(
        len(message.get("tool_calls") or ())
        for message in record.get("messages", ())
        if message.get("role") == "assistant"
    )


def expected_path(record: Mapping[str, Any]) -> str:
    expected = record.get("expected")
    if isinstance(expected, Mapping):
        return str(expected.get("path", ""))
    return ""


def final_assistant_text(record: Mapping[str, Any]) -> str:
    for message in reversed(record.get("messages", ())):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def is_sequential_focus_record(record: Mapping[str, Any]) -> bool:
    path = expected_path(record)
    return assistant_tool_call_count(record) >= 2 or (
        path == "multi_turn" and bool(record.get("expected", {}).get("requires_tool"))
    )


def is_credential_safe_clarification_record(record: Mapping[str, Any]) -> bool:
    if expected_path(record) != "clarification":
        return False
    text = final_assistant_text(record).lower()
    blocked = ("account number", "customer id", "password", " pin", "ssn")
    return "last four digits" in text and not any(token in text for token in blocked)


def is_regression_record(record: Mapping[str, Any]) -> bool:
    return expected_path(record) in {
        "tool_success",
        "tool_error",
        "no_tool_banking_faq",
        "ood",
        "hard_negative",
    }


def build_continuation_mix(
    records: Sequence[dict[str, Any]],
    *,
    sequential_multiplier: int,
    clarification_multiplier: int,
    seed: int = TRAINING_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sequential_multiplier < 1 or clarification_multiplier < 1:
        raise ValueError("continuation multipliers must be >= 1")
    mixed: list[dict[str, Any]] = []
    stats = {
        "input_records": len(records),
        "sequential_focus_records": 0,
        "credential_safe_clarification_records": 0,
        "regression_records": 0,
        "total_weighted_records": 0,
        "sequential_multiplier": sequential_multiplier,
        "clarification_multiplier": clarification_multiplier,
    }
    for record in records:
        sequential = is_sequential_focus_record(record)
        clarification = is_credential_safe_clarification_record(record)
        regression = is_regression_record(record)
        stats["sequential_focus_records"] += int(sequential)
        stats["credential_safe_clarification_records"] += int(clarification)
        stats["regression_records"] += int(regression)
        weight = 1
        if sequential:
            weight = max(weight, sequential_multiplier)
        if clarification:
            weight = max(weight, clarification_multiplier)
        mixed.extend([record] * weight)
    random.Random(seed).shuffle(mixed)
    stats["total_weighted_records"] = len(mixed)
    return mixed, stats


def dataset_identity(manifest_path: Path) -> dict[str, str | None]:
    return {
        "repository": os.environ.get("RETAIL_BANK_TOOL_SFT_DATASET_REPO"),
        "revision": os.environ.get("RETAIL_BANK_TOOL_SFT_DATASET_REVISION"),
        "manifest_sha256": sha256_file(manifest_path),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dry_run_plan(config: ContinuationConfig) -> dict[str, Any]:
    return {
        "worker": "cloud_continue_tool_sft",
        "mode": "dry_run" if config.dry_run else "execution_requested",
        "source_model_repo": config.source_model_repo,
        "source_model_revision": config.source_model_revision,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "manifest": str(config.manifest),
        "output_dir": str(config.output_dir),
        "hub_dest": config.hub_dest,
        "training": {
            "max_steps": config.max_steps,
            "max_train_seconds": config.max_train_seconds,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_seq_len": config.max_seq_len,
            "learning_rate": config.learning_rate,
            "checkpoint_every": config.checkpoint_every,
            "sequential_multiplier": config.sequential_multiplier,
            "clarification_multiplier": config.clarification_multiplier,
            "retained_regression_mix": [
                "single-tool",
                "tool-error",
                "FAQ",
                "OOD",
                "hard-negative",
            ],
        },
        "release": {
            "merge": "existing FP32 accumulation, FP16 saved weights",
            "parity": "existing merge parity report plus finalizer thresholds",
            "push_to_hub": config.push_to_hub,
        },
        "remote_guard": {
            "requires_flag": "--allow-remote-execution",
            "requires_execution_switch": "--execute-remote",
            "requires_env": f"{REMOTE_CONFIRMATION_ENV}={REMOTE_CONFIRMATION_VALUE}",
            "currently_allowed": remote_execution_allowed(config),
        },
    }


def build_training_args(config: ContinuationConfig) -> Any:
    from trl import SFTConfig  # type: ignore[import-not-found]

    return SFTConfig(
        output_dir=str(config.output_dir),
        max_steps=config.max_steps,
        max_length=config.max_seq_len,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=max(1, round(config.max_steps * 0.03)),
        bf16=True,
        tf32=tf32_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        assistant_only_loss=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,
        logging_steps=10,
        save_steps=config.checkpoint_every,
        eval_strategy="steps",
        eval_steps=max(config.checkpoint_every, min(config.max_steps, 200)),
        save_total_limit=2,
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        report_to="trackio" if config.trackio_project else [],
        project=config.trackio_project or "huggingface",
        run_name=config.trackio_run_name or "granite-tool-sft-continuation",
        push_to_hub=False,
    )


def ensure_unique_release_output(config: ContinuationConfig) -> None:
    blocked = [
        config.output_dir / "adapter",
        config.output_dir / "merged-fp16",
        config.output_dir / "continuation_training_result.json",
    ]
    existing = [str(path) for path in blocked if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite existing continuation artifacts: {existing}")


def snapshot_retained_adapter(config: ContinuationConfig) -> Path:
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    require_exact_revision(config.source_model_revision, field="--source-model-revision")
    snapshot = Path(
        snapshot_download(
            repo_id=config.source_model_repo,
            repo_type="model",
            revision=config.source_model_revision,
            allow_patterns=[
                "adapter/*",
                "training_metadata.json",
                "training_result.json",
                "merge_parity_diagnostics.json",
            ],
            token=os.environ.get("HF_TOKEN"),
        )
    )
    adapter_dir = snapshot / "adapter"
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"retained adapter is unavailable in {config.source_model_repo}")
    return snapshot


def continuation_fingerprint(
    config: ContinuationConfig,
    adapter: ToolWireAdapter,
    retained_snapshot: Path,
    mix_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract": "banking-v3-continuation-sft/v1",
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "family": config.family,
        "source_model": {
            "repository": config.source_model_repo,
            "revision": config.source_model_revision,
            "retained_adapter_sha256": sha256(
                retained_snapshot / "adapter" / "adapter_model.safetensors"
            ),
        },
        "dataset_identity": dataset_identity(config.manifest),
        "template_hash": adapter.template_hash,
        "training_seed": TRAINING_SEED,
        "continuation": {
            "max_steps": config.max_steps,
            "max_train_seconds": config.max_train_seconds,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_seq_len": config.max_seq_len,
            "sampling": dict(mix_report),
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_release_tools(config: ContinuationConfig) -> dict[str, Any]:
    remerge_script = SCRIPT_DIR / "hf_job_remerge_tool_sft.py"
    parity_script = SCRIPT_DIR / "hf_job_merge_parity.py"
    subprocess.run(
        [
            sys.executable,
            str(remerge_script),
            "--output-root",
            str(config.output_dir),
            "--output-subdir",
            "merged-fp16",
            "--base-model",
            config.base_model,
            "--base-revision",
            config.base_revision,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(parity_script),
            "--output-root",
            str(config.output_dir),
            "--merged-subdir",
            "merged-fp16",
            "--base-model",
            config.base_model,
            "--base-revision",
            config.base_revision,
            "--inference-dtype",
            "float16",
        ],
        check=True,
    )
    parity_path = config.output_dir / "merge_parity_diagnostics_merged-fp16_float16.json"
    report = json.loads(parity_path.read_text(encoding="utf-8"))
    metrics = validate_parity(
        report,
        minimum_argmax_agreement=0.999,
        maximum_logit_difference=0.3,
        maximum_p999_difference=0.07,
    )
    return {
        "remerge_report": str(config.output_dir / "fp16_remerge.json"),
        "parity_report": str(parity_path),
        "parity_metrics": metrics,
    }


def upload_release(
    config: ContinuationConfig,
    *,
    result_path: Path,
    metadata_path: Path,
) -> str:
    from huggingface_hub import HfApi  # type: ignore[import-not-found]

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(config.hub_dest, repo_type="model", private=False, exist_ok=True)
    weights_commit = api.upload_folder(
        repo_id=config.hub_dest,
        repo_type="model",
        folder_path=config.output_dir / "merged-fp16",
        allow_patterns=list(MERGED_ALLOWLIST),
        commit_message="Publish continuation SFT merged weights",
    )
    weights_revision = str(weights_commit.oid)
    require_exact_revision(weights_revision, field="weights revision")
    api.upload_folder(
        repo_id=config.hub_dest,
        repo_type="model",
        folder_path=config.output_dir / "adapter",
        path_in_repo="adapter",
        allow_patterns=list(ADAPTER_ALLOWLIST),
        commit_message="Retain continuation SFT adapter",
    )
    api.upload_file(
        repo_id=config.hub_dest,
        repo_type="model",
        path_or_fileobj=result_path,
        path_in_repo="training_result.json",
        commit_message="Record continuation SFT result",
    )
    api.upload_file(
        repo_id=config.hub_dest,
        repo_type="model",
        path_or_fileobj=metadata_path,
        path_in_repo="training_metadata.json",
        commit_message="Record continuation SFT metadata",
    )
    api.upload_file(
        repo_id=config.hub_dest,
        repo_type="model",
        path_or_fileobj=config.output_dir / "merge_parity_diagnostics_merged-fp16_float16.json",
        path_in_repo="merge_parity_diagnostics.json",
        commit_message="Record continuation merge parity diagnostics",
    )
    api.upload_file(
        repo_id=config.hub_dest,
        repo_type="model",
        path_or_fileobj=config.output_dir / "fp16_remerge.json",
        path_in_repo="fp16_remerge.json",
        commit_message="Record continuation FP16 remerge provenance",
    )
    return weights_revision


def run_remote_continuation(config: ContinuationConfig) -> dict[str, Any]:
    assert_remote_execution_allowed(config)
    require_exact_revision(config.source_model_revision, field="--source-model-revision")
    ensure_unique_release_output(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    seed_training(TRAINING_SEED)

    from datasets import Dataset as HfDataset  # type: ignore[import-not-found]
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTTrainer  # type: ignore[import-not-found]

    retained_snapshot = snapshot_retained_adapter(config)
    retained_adapter_dir = retained_snapshot / "adapter"
    tokenizer = AutoTokenizer.from_pretrained(retained_adapter_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter = ToolWireAdapter(
        tokenizer,
        family=config.family,
        public_tool_manifest=PUBLIC_BANKING_TOOL_MANIFEST,
        pad_to_max_length=False,
    )
    train_records = load_manifest_records(config.manifest, "train")
    validation_records = load_manifest_records(config.manifest, "validation")
    mixed_train_records, mix_report = build_continuation_mix(
        train_records,
        sequential_multiplier=config.sequential_multiplier,
        clarification_multiplier=config.clarification_multiplier,
    )
    train_examples = tokenize_records(
        mixed_train_records,
        adapter,
        max_seq_len=config.max_seq_len,
    )
    validation_examples = tokenize_records(
        validation_records,
        adapter,
        max_seq_len=config.max_seq_len,
    )
    train_dataset = HfDataset.from_dict(
        {
            name: [example[name].tolist() for example in train_examples]
            for name in ("input_ids", "attention_mask", "labels")
        }
    )
    validation_dataset = HfDataset.from_dict(
        {
            name: [example[name].tolist() for example in validation_examples]
            for name in ("input_ids", "attention_mask", "labels")
        }
    )
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        revision=config.base_revision,
        dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
    )
    model = PeftModel.from_pretrained(base, retained_adapter_dir, is_trainable=True)
    model.enable_input_require_grads()

    class WallClockStopCallback(TrainerCallback):
        def __init__(self, limit_seconds: int) -> None:
            self.limit_seconds = limit_seconds
            self.started_at = 0.0

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, state, control, kwargs
            self.started_at = time.monotonic()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state, kwargs
            if time.monotonic() - self.started_at >= self.limit_seconds:
                control.should_training_stop = True
                control.should_save = True
            return control

    fingerprint = continuation_fingerprint(config, adapter, retained_snapshot, mix_report)
    trainer = SFTTrainer(
        model=model,
        args=build_training_args(config),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=partial(
            collate_pretokenized,
            pad_token_id=int(tokenizer.pad_token_id),
        ),
        processing_class=tokenizer,
        callbacks=[WallClockStopCallback(config.max_train_seconds)],
    )
    train_output = trainer.train()
    if config.trackio_project:
        from transformers.integrations import TrackioCallback  # type: ignore[import-not-found]

        trainer.remove_callback(TrackioCallback)
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(config.output_dir / "adapter"))
    tokenizer.save_pretrained(config.output_dir / "adapter")
    actual_step = int(trainer.state.global_step)
    metadata = {
        "contract": "banking-v3-continuation-sft-metadata/v1",
        "step": actual_step,
        "created_at_unix": int(time.time()),
        "worker": "cloud_continue_tool_sft",
        "fingerprint": fingerprint,
        "train_metrics": dict(train_output.metrics),
        "eval_metrics": dict(eval_metrics),
    }
    metadata_path = config.output_dir / "continuation_training_metadata.json"
    write_json(metadata_path, metadata)
    del trainer
    del model
    del base
    gc.collect()
    torch.cuda.empty_cache()

    release = run_release_tools(config)
    result = {
        "contract": "banking-v3-continuation-sft-result/v1",
        "worker": "cloud_continue_tool_sft",
        "steps": actual_step,
        "source_model_repo": config.source_model_repo,
        "source_model_revision": config.source_model_revision,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "dataset_identity": dataset_identity(config.manifest),
        "template_hash": adapter.template_hash,
        "sampling": mix_report,
        "train_metrics": dict(train_output.metrics),
        "eval_metrics": dict(eval_metrics),
        "release": release,
        "adapter_sha256": sha256(config.output_dir / "adapter" / "adapter_model.safetensors"),
        "merged_model_sha256": sha256(config.output_dir / "merged-fp16" / "model.safetensors"),
        "pushed_to_hub": False,
    }
    result_path = config.output_dir / "continuation_training_result.json"
    write_json(result_path, result)
    if config.push_to_hub:
        weights_revision = upload_release(
            config,
            result_path=result_path,
            metadata_path=metadata_path,
        )
        result["weights_revision"] = weights_revision
        result["pushed_to_hub"] = config.hub_dest
        write_json(result_path, result)
        from huggingface_hub import HfApi  # type: ignore[import-not-found]

        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            repo_id=config.hub_dest,
            repo_type="model",
            path_or_fileobj=result_path,
            path_in_repo="training_result.json",
            commit_message="Record continuation SFT published revision",
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(parse_args(argv))
    require_exact_revision(config.source_model_revision, field="--source-model-revision")
    if config.max_steps < 1 or config.max_train_seconds < 60:
        raise ValueError("continuation caps must allow at least one step and 60 seconds")
    if config.dry_run:
        print(json.dumps(build_dry_run_plan(config), indent=2, sort_keys=True))
    else:
        print(json.dumps(run_remote_continuation(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
