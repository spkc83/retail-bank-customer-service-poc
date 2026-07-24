from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from hello_slm.artifacts import atomic_write_json, environment_record, sha256_file
from hello_slm.config import ExperimentConfig
from hello_slm.training import (
    _resolve_precision_runtime,
    load_model_from_checkpoint,
    load_split,
    split_metrics,
)


def evaluate(
    config: ExperimentConfig,
    *,
    checkpoint_path: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    started = time.time()
    precision_runtime = _resolve_precision_runtime(config)
    device = torch.device(precision_runtime.device)
    model, checkpoint, context = load_model_from_checkpoint(config, checkpoint_path, device=device)
    metrics = split_metrics(
        model,
        load_split(config, split),
        device=device,
        precision_runtime=precision_runtime,
    )
    is_smoke = config.data["run"]["id"] == "smoke"
    report = {
        "command": "eval",
        "status": "success",
        "profile": config.data["run"]["id"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "global_step": checkpoint["global_step"],
        "heldout_split": split,
        "heldout_loss": metrics["loss"],
        "heldout_perplexity": metrics["perplexity"],
        "assistant_token_accuracy": metrics["assistant_token_accuracy"],
        "loss_tokens": metrics["loss_tokens"],
        "release_eligible": False if is_smoke else bool(metrics["perplexity"] <= 35.0),
        "precision_runtime": precision_runtime.to_mapping(),
        "fingerprints": context,
        "environment": environment_record(),
        "duration_seconds": time.time() - started,
    }
    atomic_write_json(config.artifact_dir / "reports" / "eval.json", report)
    return report
