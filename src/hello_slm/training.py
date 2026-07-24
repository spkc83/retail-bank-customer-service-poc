from __future__ import annotations

import copy
import json
import math
import random
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from hello_slm.artifacts import (
    atomic_torch_save,
    atomic_write_json,
    capture_rng_state,
    environment_record,
    restore_rng_state,
    sha256_file,
)
from hello_slm.config import ExperimentConfig, canonical_sha256, load_experiment_config
from hello_slm.data import (
    build_dataset_artifacts,
    conversations_for_tokenizer,
    load_and_validate_corpus,
)
from hello_slm.model import HelloSLMModel, ModelConfig
from hello_slm.tokenizer import SPECIAL_TOKENS, load_tokenizer, train_restricted_bpe

CHAT_TEMPLATE_HASH = canonical_sha256(
    {"format_version": 1, "template": "restricted-render-chat-v1", "special_tokens": SPECIAL_TOKENS}
)


class PipelineError(RuntimeError):
    """Expected closed pipeline failure."""


@dataclass(frozen=True)
class PrecisionRuntime:
    requested_device: str
    requested_precision: str
    device: str
    precision: str
    autocast_enabled: bool
    grad_scaler_enabled: bool
    fallback_applied: bool
    fallback_reason: str | None = None

    @property
    def torch_dtype(self) -> torch.dtype | None:
        if self.precision == "float16":
            return torch.float16
        if self.precision == "bfloat16":
            return torch.bfloat16
        return None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "requested_precision": self.requested_precision,
            "device": self.device,
            "precision": self.precision,
            "autocast_enabled": self.autocast_enabled,
            "grad_scaler_enabled": self.grad_scaler_enabled,
            "fallback_applied": self.fallback_applied,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class TrainingState:
    global_step: int
    dataloader_cursor: int
    consumed_examples: int
    consumed_total_tokens: int
    consumed_loss_tokens: int

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> TrainingState:
        return cls(
            global_step=int(data["global_step"]),
            dataloader_cursor=int(data["dataloader_cursor"]),
            consumed_examples=int(data["consumed_examples"]),
            consumed_total_tokens=int(data["consumed_total_tokens"]),
            consumed_loss_tokens=int(data["consumed_loss_tokens"]),
        )

    def to_mapping(self) -> dict[str, int]:
        return {
            "global_step": self.global_step,
            "dataloader_cursor": self.dataloader_cursor,
            "consumed_examples": self.consumed_examples,
            "consumed_total_tokens": self.consumed_total_tokens,
            "consumed_loss_tokens": self.consumed_loss_tokens,
        }


def load_config(config_path: str | Path, work_dir: str | Path | None = None) -> ExperimentConfig:
    config = load_experiment_config(config_path)
    if work_dir is None:
        return config
    data = copy.deepcopy(config.data)
    data["run"]["artifact_dir"] = str(Path(work_dir).resolve())
    return replace(config, data=data, effective_hash=canonical_sha256(data))


def validate_run(config: ExperimentConfig, *, structural: bool = False) -> dict[str, Any]:
    started = time.time()
    report = {
        "command": "validate",
        "status": "success",
        "structural_only": structural,
        "effective_config_hash": config.effective_hash,
        "parameter_count": config.parameter_count,
        "environment": environment_record(),
        "duration_seconds": time.time() - started,
    }
    if not structural:
        corpus = load_and_validate_corpus(config)
        report.update(
            {
                "corpus_manifest_hash": corpus.manifest_hash,
                "corpus_fingerprint": corpus.fingerprint,
                "corpus_report": corpus.report,
            }
        )
    atomic_write_json(config.artifact_dir / "reports" / "validate.json", report)
    atomic_write_json(config.artifact_dir / "effective-config.json", config.data)
    return report


def build_tokenizer(config: ExperimentConfig) -> dict[str, Any]:
    started = time.time()
    _seed_everything(int(config.data["seeds"]["tokenizer"]))
    corpus = load_and_validate_corpus(config)
    tokenizer = train_restricted_bpe(
        conversations_for_tokenizer(corpus),
        vocab_size=int(config.data["tokenizer"]["vocab_size"]),
        min_frequency=int(config.data["tokenizer"]["min_frequency"]),
        corpus_manifest_hash=corpus.manifest_hash,
        corpus_fingerprint=corpus.fingerprint,
        tokenizer_config=config.data["tokenizer"],
    )
    tokenizer_path = config.artifact_dir / "tokenizer" / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    report = {
        "command": "build-tokenizer",
        "status": "success",
        "effective_config_hash": config.effective_hash,
        "corpus_manifest_hash": corpus.manifest_hash,
        "corpus_fingerprint": corpus.fingerprint,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_fingerprint": tokenizer.to_artifact()["tokenizer_fingerprint"],
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "environment": environment_record(),
        "duration_seconds": time.time() - started,
    }
    atomic_write_json(config.artifact_dir / "reports" / "tokenizer.json", report)
    return report


def build_dataset(config: ExperimentConfig) -> dict[str, Any]:
    started = time.time()
    _seed_everything(int(config.data["seeds"]["dataset"]))
    corpus = load_and_validate_corpus(config)
    tokenizer = load_tokenizer(config.artifact_dir / "tokenizer" / "tokenizer.json")
    manifest = build_dataset_artifacts(config, corpus, tokenizer)
    report = {
        "command": "build-dataset",
        "status": "success",
        "effective_config_hash": config.effective_hash,
        "corpus_manifest_hash": corpus.manifest_hash,
        "corpus_fingerprint": corpus.fingerprint,
        "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "dataset_manifest": manifest,
        "environment": environment_record(),
        "duration_seconds": time.time() - started,
    }
    atomic_write_json(config.artifact_dir / "reports" / "dataset.json", report)
    return report


def train(
    config: ExperimentConfig,
    *,
    max_steps: int | None = None,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    precision_runtime = _resolve_precision_runtime(config)
    device = torch.device(precision_runtime.device)
    _seed_everything(int(config.data["seeds"]["model"]))
    context = artifact_context(config)
    total_steps = int(max_steps or config.data["training"]["total_steps"])
    if total_steps < 1:
        raise PipelineError("max_steps must be positive")

    model = HelloSLMModel(ModelConfig.from_mapping(config.data["model"])).to(device)
    optimizer = _build_optimizer(model, config)
    scheduler = WarmupCosineScheduler(optimizer, config)
    scaler = _build_grad_scaler(precision_runtime)
    state = TrainingState(0, 0, 0, 0, 0)

    if resume is not None:
        checkpoint = _load_checkpoint(resume, device)
        _validate_checkpoint(
            checkpoint,
            config,
            context,
            model,
            purpose="resume",
            precision_runtime=precision_runtime,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "grad_scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        state = TrainingState.from_mapping(checkpoint["training_state"])
        if state.global_step > total_steps:
            raise PipelineError("resume checkpoint is already past requested max_steps")

    train_split = load_split(config, "train")
    validation_split = load_split(config, "validation")
    if train_split["input_ids"].numel() == 0:
        raise PipelineError("train split is empty")
    order = _deterministic_order(
        int(train_split["input_ids"].shape[0]), int(config.data["seeds"]["dataloader"])
    )

    last_loss = math.nan
    last_grad_norm = math.nan
    last_lr = scheduler.current_lr
    validation_history: list[dict[str, Any]] = []
    if config.data["evaluation"]["eval_at_step_zero"] and state.global_step == 0:
        validation_history.append(
            _validation_record(
                model, validation_split, device, state, last_lr, precision_runtime
            )
        )
    model.train()
    while state.global_step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(int(config.data["training"]["gradient_accumulation_steps"])):
            batch, state = _next_batch(config, train_split, order, state, device)
            with _autocast_context(precision_runtime):
                logits, _ = model(batch["input_ids"])
                loss = F.cross_entropy(
                    logits.reshape(-1, model.config.vocab_size),
                    batch["labels"].reshape(-1),
                    ignore_index=-100,
                )
            if not torch.isfinite(loss):
                raise PipelineError("non-finite training loss")
            scaled_loss = loss / int(config.data["training"]["gradient_accumulation_steps"])
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach().cpu())
        scaler.unscale_(optimizer)
        _reject_nonfinite_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.data["training"]["max_grad_norm"])
        )
        if not torch.isfinite(grad_norm):
            raise PipelineError("non-finite gradient norm")
        scheduler.step(state.global_step)
        scaler.step(optimizer)
        scaler.update()
        state = replace(state, global_step=state.global_step + 1)
        last_loss = accumulated_loss / int(config.data["training"]["gradient_accumulation_steps"])
        last_grad_norm = float(grad_norm.detach().cpu())
        last_lr = scheduler.current_lr
        eval_interval = int(config.data["evaluation"]["interval_steps"])
        if state.global_step % eval_interval == 0 or state.global_step == total_steps:
            validation_history.append(
                _validation_record(
                    model, validation_split, device, state, last_lr, precision_runtime
                )
            )
            model.train()
        interval = int(config.data["checkpointing"]["interval_steps"])
        if state.global_step % interval == 0 or state.global_step == total_steps:
            _write_checkpoint(
                config,
                context,
                model,
                optimizer,
                scheduler,
                scaler,
                state,
                precision_runtime,
            )

    latest = config.artifact_dir / "checkpoints" / "latest.pt"
    report = {
        "command": "train",
        "status": "success",
        "effective_config_hash": config.effective_hash,
        "global_step": state.global_step,
        "train_loss": last_loss,
        "learning_rate": last_lr,
        "gradient_norm": last_grad_norm,
        "validation": validation_history[-1] if validation_history else None,
        "validation_history": validation_history,
        "checkpoint": str(latest),
        "checkpoint_sha256": sha256_file(latest),
        "training_state": state.to_mapping(),
        "precision_runtime": precision_runtime.to_mapping(),
        "fingerprints": context,
        "environment": environment_record(),
        "duration_seconds": time.time() - started,
    }
    atomic_write_json(config.artifact_dir / "reports" / "train.json", report)
    return report


def artifact_context(config: ExperimentConfig) -> dict[str, Any]:
    corpus = load_and_validate_corpus(config)
    tokenizer_path = config.artifact_dir / "tokenizer" / "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    dataset_manifest = load_dataset_manifest(config)
    return {
        "effective_config_hash": config.effective_hash,
        "parameter_count": config.parameter_count,
        "model_config_hash": canonical_sha256(config.data["model"]),
        "corpus_manifest_hash": corpus.manifest_hash,
        "corpus_fingerprint": corpus.fingerprint,
        "tokenizer_fingerprint": tokenizer.to_artifact()["tokenizer_fingerprint"],
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
        "dataset_splits": dataset_manifest["splits"],
        "special_tokens": SPECIAL_TOKENS,
        "chat_template_hash": CHAT_TEMPLATE_HASH,
    }


def load_dataset_manifest(config: ExperimentConfig) -> dict[str, Any]:
    path = config.artifact_dir / "dataset" / "manifest.json"
    if not path.exists():
        raise PipelineError(f"dataset manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(config: ExperimentConfig, split: str) -> dict[str, Any]:
    manifest = load_dataset_manifest(config)
    record = manifest["splits"][split]
    path = Path(record["path"]).resolve()
    dataset_root = (config.artifact_dir / "dataset").resolve()
    if not path.is_relative_to(dataset_root):
        raise PipelineError(f"dataset shard escapes artifact directory: {path}")
    if sha256_file(path) != record["sha256"]:
        raise PipelineError(f"dataset shard digest mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model_from_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    *,
    device: torch.device | None = None,
) -> tuple[HelloSLMModel, dict[str, Any], dict[str, Any]]:
    device = device or torch.device("cpu")
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model = HelloSLMModel(ModelConfig.from_mapping(config.data["model"])).to(device)
    context = artifact_context(config)
    _validate_checkpoint(checkpoint, config, context, model, purpose="evaluation")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, context


def _validation_record(
    model: HelloSLMModel,
    validation_split: dict[str, Any],
    device: torch.device,
    state: TrainingState,
    learning_rate: float,
    precision_runtime: PrecisionRuntime | None = None,
) -> dict[str, Any]:
    metrics = split_metrics(
        model, validation_split, device=device, precision_runtime=precision_runtime
    )
    return {
        "global_step": state.global_step,
        "consumed_loss_tokens": state.consumed_loss_tokens,
        "learning_rate": learning_rate,
        "validation_loss": metrics["loss"],
        "validation_perplexity": metrics["perplexity"],
        "assistant_token_accuracy": metrics["assistant_token_accuracy"],
        "loss_tokens": metrics["loss_tokens"],
    }


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, config: ExperimentConfig) -> None:
        self.optimizer = optimizer
        self.max_lr = float(config.data["training"]["optimizer"]["learning_rate"])
        self.min_lr = float(config.data["training"]["scheduler"]["min_learning_rate"])
        self.warmup_steps = int(config.data["training"]["scheduler"]["warmup_steps"])
        self.total_steps = int(config.data["training"]["total_steps"])
        self.current_lr = self.max_lr
        self.last_scheduled_step = -1
        self._set_lr(self.max_lr)

    def step(self, step: int) -> None:
        if self.warmup_steps and step < self.warmup_steps:
            lr = self.max_lr * (step + 1) / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        self._set_lr(lr)
        self.last_scheduled_step = step

    def state_dict(self) -> dict[str, Any]:
        return {"current_lr": self.current_lr, "last_scheduled_step": self.last_scheduled_step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.current_lr = float(state["current_lr"])
        self.last_scheduled_step = int(state["last_scheduled_step"])
        self._set_lr(self.current_lr)

    def _set_lr(self, lr: float) -> None:
        self.current_lr = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_lr


@torch.no_grad()
def split_metrics(
    model: HelloSLMModel,
    split_payload: dict[str, Any],
    *,
    device: torch.device,
    precision_runtime: PrecisionRuntime | None = None,
) -> dict[str, float | int]:
    model.eval()
    losses: list[torch.Tensor] = []
    correct = 0
    total = 0
    for start in range(0, int(split_payload["input_ids"].shape[0]), 16):
        input_ids = split_payload["input_ids"][start : start + 16].to(device)
        labels = split_payload["labels"][start : start + 16].to(device)
        if labels.numel() == 0 or (labels != -100).sum().item() == 0:
            continue
        with _autocast_context(precision_runtime):
            logits, _ = model(input_ids)
        losses.append(
            F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            ).detach()
        )
        mask = labels != -100
        predictions = logits.argmax(dim=-1)
        correct += int((predictions[mask] == labels[mask]).sum().item())
        total += int(mask.sum().item())
    if not losses or total == 0:
        raise PipelineError("evaluation split has no loss-bearing tokens")
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise PipelineError("non-finite evaluation loss")
    loss_value = float(loss.cpu())
    return {
        "loss": loss_value,
        "perplexity": math.exp(loss_value) if loss_value < math.log(1e9) else 1e9,
        "assistant_token_accuracy": correct / total,
        "loss_tokens": total,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _resolve_precision_runtime(config: ExperimentConfig) -> PrecisionRuntime:
    training = config.data["training"]
    requested_device = str(training["device"])
    requested_precision = str(training["precision"])
    allow_fallback = bool(training["allow_precision_fallback"])
    if requested_precision not in {"float32", "float16", "bfloat16"}:
        raise PipelineError(f"unsupported training precision: {requested_precision}")
    if requested_device not in {"cpu", "cuda"}:
        raise PipelineError(f"unsupported training device: {requested_device}")

    device = requested_device
    precision = requested_precision
    fallback_reason: str | None = None
    if device == "cuda" and not torch.cuda.is_available():
        if not allow_fallback:
            raise PipelineError("CUDA was requested but is unavailable")
        device = "cpu"
        precision = "float32"
        fallback_reason = "CUDA was requested but is unavailable"
    if device == "cpu" and precision != "float32":
        if not allow_fallback:
            raise PipelineError("CPU training requires float32 precision")
        precision = "float32"
        fallback_reason = "CPU training requires float32 precision"
    if device == "cuda" and precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        if not allow_fallback:
            raise PipelineError("CUDA bfloat16 precision was requested but is unavailable")
        precision = "float16"
        fallback_reason = "CUDA bfloat16 precision was requested but is unavailable"
    autocast_enabled = device == "cuda" and precision in {"float16", "bfloat16"}
    grad_scaler_enabled = device == "cuda" and precision == "float16"
    return PrecisionRuntime(
        requested_device=requested_device,
        requested_precision=requested_precision,
        device=device,
        precision=precision,
        autocast_enabled=autocast_enabled,
        grad_scaler_enabled=grad_scaler_enabled,
        fallback_applied=fallback_reason is not None,
        fallback_reason=fallback_reason,
    )


def _autocast_context(precision_runtime: PrecisionRuntime | None) -> AbstractContextManager[Any]:
    if precision_runtime is None or not precision_runtime.autocast_enabled:
        return nullcontext()
    dtype = precision_runtime.torch_dtype
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=precision_runtime.device, dtype=dtype)


def _build_grad_scaler(precision_runtime: PrecisionRuntime) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        device=precision_runtime.device,
        enabled=precision_runtime.grad_scaler_enabled,
    )


def _build_optimizer(model: HelloSLMModel, config: ExperimentConfig) -> torch.optim.AdamW:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("weight") and "norm" not in name:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    opt_config = config.data["training"]["optimizer"]
    betas = (float(opt_config["betas"][0]), float(opt_config["betas"][1]))
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(opt_config["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(opt_config["learning_rate"]),
        betas=betas,
        eps=float(opt_config["eps"]),
    )


def _deterministic_order(length: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator).tolist()


def _next_batch(
    config: ExperimentConfig,
    split_payload: dict[str, Any],
    order: list[int],
    state: TrainingState,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], TrainingState]:
    batch_size = int(config.data["training"]["micro_batch_size"])
    indices = []
    cursor = state.dataloader_cursor
    for _ in range(batch_size):
        indices.append(order[cursor % len(order)])
        cursor += 1
    input_ids = split_payload["input_ids"][indices].to(device)
    labels = split_payload["labels"][indices].to(device)
    attention = split_payload["attention_mask"][indices].to(device)
    return (
        {"input_ids": input_ids, "labels": labels, "attention_mask": attention},
        replace(
            state,
            dataloader_cursor=cursor,
            consumed_examples=state.consumed_examples + len(indices),
            consumed_total_tokens=state.consumed_total_tokens + int(attention.sum().item()),
            consumed_loss_tokens=state.consumed_loss_tokens + int((labels != -100).sum().item()),
        ),
    )


def _reject_nonfinite_gradients(model: HelloSLMModel) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise PipelineError("non-finite gradient")


def _write_checkpoint(
    config: ExperimentConfig,
    context: dict[str, Any],
    model: HelloSLMModel,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    state: TrainingState,
    precision_runtime: PrecisionRuntime,
) -> None:
    checkpoint_dir = config.artifact_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / f"step-{state.global_step:08d}.pt"
    payload = {
        "format_version": 1,
        "model_config": config.data["model"],
        "parameter_count": model.parameter_count,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "global_step": state.global_step,
        "training_state": state.to_mapping(),
        "precision_runtime": precision_runtime.to_mapping(),
        "rng_state": capture_rng_state(),
        "fingerprints": context,
        "environment": environment_record(),
    }
    atomic_torch_save(checkpoint_path, payload)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    atomic_write_json(
        checkpoint_path.with_suffix(checkpoint_path.suffix + ".manifest.json"),
        {
            "format_version": 1,
            "sha256": checkpoint_sha256,
            "bytes": checkpoint_path.stat().st_size,
        },
    )
    payload["checkpoint_sha256"] = checkpoint_sha256
    payload["checkpoint_path"] = str(checkpoint_path)
    latest_path = checkpoint_dir / "latest.pt"
    atomic_torch_save(latest_path, payload)
    atomic_write_json(
        latest_path.with_suffix(latest_path.suffix + ".manifest.json"),
        {
            "format_version": 1,
            "sha256": sha256_file(latest_path),
            "bytes": latest_path.stat().st_size,
        },
    )
    _prune_checkpoints(checkpoint_dir, int(config.data["checkpointing"]["keep_last"]))


def _prune_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    checkpoints = sorted(checkpoint_dir.glob("step-*.pt"))
    for path in checkpoints[:-keep_last]:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".manifest.json").unlink(missing_ok=True)


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path).resolve()
    manifest_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise PipelineError(f"checkpoint integrity manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("format_version", 0)) != 1:
        raise PipelineError("unsupported checkpoint integrity manifest")
    if checkpoint_path.stat().st_size != int(manifest["bytes"]):
        raise PipelineError("checkpoint byte-count mismatch")
    if sha256_file(checkpoint_path) != manifest["sha256"]:
        raise PipelineError("checkpoint digest mismatch")
    return torch.load(checkpoint_path, map_location=device, weights_only=True)


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    config: ExperimentConfig,
    context: dict[str, Any],
    model: HelloSLMModel,
    *,
    purpose: str,
    precision_runtime: PrecisionRuntime | None = None,
) -> None:
    required = {
        "format_version",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "global_step",
        "training_state",
        "rng_state",
        "fingerprints",
        "parameter_count",
        "model_config",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise PipelineError(f"checkpoint missing required state: {missing}")
    if int(checkpoint["format_version"]) != 1:
        raise PipelineError("unsupported checkpoint format_version")
    if int(checkpoint["parameter_count"]) != model.parameter_count:
        raise PipelineError("checkpoint parameter-count mismatch")
    stored = checkpoint["fingerprints"]
    for key in (
        "effective_config_hash",
        "corpus_manifest_hash",
        "corpus_fingerprint",
        "tokenizer_fingerprint",
        "dataset_fingerprint",
        "chat_template_hash",
        "model_config_hash",
    ):
        if stored.get(key) != context.get(key):
            raise PipelineError(f"{purpose} rejected {key} mismatch")
    if stored.get("special_tokens") != SPECIAL_TOKENS:
        raise PipelineError("checkpoint special-token IDs mismatch")
    if checkpoint["model_config"].get("vocab_size") != config.data["model"]["vocab_size"]:
        raise PipelineError("checkpoint vocab_size mismatch")
    if checkpoint["model_config"].get("max_seq_len") != config.data["model"]["max_seq_len"]:
        raise PipelineError("checkpoint max_seq_len mismatch")
    if precision_runtime is not None:
        stored_runtime = checkpoint.get("precision_runtime")
        if stored_runtime != precision_runtime.to_mapping():
            raise PipelineError(f"{purpose} rejected precision runtime mismatch")
