#!/usr/bin/env python
"""Executable banking-v2 MoE training worker.

Default behavior is safe: dry-run prints the resolved plan. Local tests use
``--run-tiny-smoke`` with a tiny randomly initialized MoE. The full remote path
loads the pinned dense Qwen checkpoint, converts it to banking-v2 Qwen2-MoE, and
trains on the prepared banking-v2 manifest only after both an explicit flag and
environment confirmation are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from hello_slm.banking_moe import (
    BANKING_V2_BASE_MODEL,
    BANKING_V2_BASE_REVISION,
    BANKING_V2_GENERATIVE_DATASET,
    BANKING_V2_HUB_DEST,
    BANKING_V2_OOD_STOCK_RESPONSE,
    BANKING_V2_TOTAL_PARAMETERS,
    BankingV2Pins,
    apply_banking_v2_trainable_policy,
    banking_v2_qwen2_moe_config,
    banking_v2_training_summary,
    convert_dense_qwen_to_banking_moe_state,
    effective_parameter_count,
    expert_assignment_health,
    expert_health_passed,
    routed_down_grad_flags,
    tiny_qwen2_moe_config,
    topk_assignments,
)

REMOTE_CONFIRMATION_ENV = "HELLO_SLM_ALLOW_REMOTE_TRAINING"
REMOTE_CONFIRMATION_VALUE = "banking-v2"


@dataclass(frozen=True)
class WorkerConfig:
    manifest: Path
    output_dir: Path
    max_steps: int
    batch_size: int
    max_seq_len: int
    learning_rate: float
    checkpoint_every: int
    resume_from: Path | None
    dry_run: bool
    run_tiny_smoke: bool
    allow_remote_execution: bool
    push_to_hub: bool
    hub_dest: str


class TokenizedChatDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, examples: Sequence[dict[str, Tensor]]) -> None:
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self._examples[index]


class SimpleBankingTokenizer:
    """Tiny offline tokenizer for smoke tests; not used for the remote 9B path."""

    pad_token_id = 0
    eos_token_id = 2
    vocab_size = 128

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        rendered = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "\nassistant:"
        return rendered

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        **_: Any,
    ) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [max(3, min(127, ord(char))) for char in text]}

    def save_pretrained(self, path: str | Path) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "simple_tokenizer.json").write_text(
            json.dumps({"type": "simple_banking_tokenizer", "vocab_size": self.vocab_size}),
            encoding="utf-8",
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=BANKING_V2_GENERATIVE_DATASET)
    parser.add_argument("--output-dir", default="artifacts/banking-v2-moe-9b")
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume-from")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute-remote", action="store_false", dest="dry_run")
    parser.add_argument("--run-tiny-smoke", action="store_true")
    parser.add_argument("--allow-remote-execution", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-dest", default=BANKING_V2_HUB_DEST)
    return parser.parse_args(argv)


def worker_config_from_args(args: argparse.Namespace) -> WorkerConfig:
    return WorkerConfig(
        manifest=Path(args.manifest),
        output_dir=Path(args.output_dir),
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
        max_seq_len=int(args.max_seq_len),
        learning_rate=float(args.learning_rate),
        checkpoint_every=int(args.checkpoint_every),
        resume_from=Path(args.resume_from) if args.resume_from else None,
        dry_run=bool(args.dry_run),
        run_tiny_smoke=bool(args.run_tiny_smoke),
        allow_remote_execution=bool(args.allow_remote_execution),
        push_to_hub=bool(args.push_to_hub),
        hub_dest=str(args.hub_dest),
    )


def remote_execution_allowed(config: WorkerConfig) -> bool:
    return bool(
        not config.dry_run
        and config.allow_remote_execution
        and os.environ.get(REMOTE_CONFIRMATION_ENV) == REMOTE_CONFIRMATION_VALUE
    )


def assert_remote_execution_allowed(config: WorkerConfig) -> None:
    if not remote_execution_allowed(config):
        raise PermissionError(
            "Remote 9B training requires --execute-remote, --allow-remote-execution, and "
            f"{REMOTE_CONFIRMATION_ENV}={REMOTE_CONFIRMATION_VALUE}."
        )


def build_dry_run_plan(config: WorkerConfig) -> dict[str, Any]:
    return {
        "worker": "cloud_train_banking_moe",
        "mode": "dry_run" if config.dry_run else "execution_requested",
        "pins": asdict(BankingV2Pins()),
        "training_summary": banking_v2_training_summary(),
        "manifest": str(config.manifest),
        "output_dir": str(config.output_dir),
        "max_steps": config.max_steps,
        "batch_size": config.batch_size,
        "max_seq_len": config.max_seq_len,
        "checkpoint_every": config.checkpoint_every,
        "resume_from": str(config.resume_from) if config.resume_from else None,
        "remote_guard": {
            "requires_flag": "--allow-remote-execution",
            "requires_execution_switch": "--execute-remote",
            "requires_env": f"{REMOTE_CONFIRMATION_ENV}={REMOTE_CONFIRMATION_VALUE}",
            "currently_allowed": remote_execution_allowed(config),
        },
        "remote_actions_when_allowed": [
            "download pinned dense Qwen checkpoint",
            "instantiate exact banking-v2 Qwen2-MoE",
            "convert dense state dict into MoE state dict",
            "load prepared banking-v2 Bitext manifest train/validation splits",
            "train with Accelerate/FSDP-compatible BF16 loop",
            "checkpoint with resume metadata and expert-health telemetry",
            "save final model/tokenizer locally",
            "optionally push to private Hub when --push-to-hub is set",
        ],
        "will_not_do_without_guard": [
            "download 9B or dense base weights",
            "write to Hugging Face Hub",
            "create a remote repository",
            "start a paid/cloud job",
        ],
    }


def load_manifest_records(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    paths: list[Path] = []
    if "generative_sft" in manifest:
        for entry in manifest["generative_sft"]:
            if (
                entry.get("included_for_generative_sft", True)
                and entry.get("name") == split
            ):
                declared = Path(entry["path"])
                local_candidate = base_dir / declared.name
                paths.append(
                    local_candidate.resolve()
                    if local_candidate.exists()
                    else declared.resolve()
                )
    elif "entries" in manifest:
        for entry in manifest["entries"]:
            if entry.get("included", True) and entry.get("split") == split:
                paths.append((base_dir / entry["path"]).resolve())
    elif "splits" in manifest and split in manifest["splits"]:
        split_entry = manifest["splits"][split]
        value = split_entry["path"] if isinstance(split_entry, dict) else split_entry
        paths.append((base_dir / value).resolve())
    else:
        raise ValueError(f"manifest {manifest_path} does not declare split {split!r}")

    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    if not records:
        raise ValueError(f"manifest split {split!r} is empty")
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_identity(manifest_path: Path) -> dict[str, str | None]:
    report_path = manifest_path.parent / "preparation-report.json"
    source_lock_path = manifest_path.parent.parent / "sources" / "banking-v2.lock.json"
    corpus_fingerprint = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        corpus_fingerprint = str(report["summary"]["corpus_fingerprint"])
    return {
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "corpus_fingerprint": corpus_fingerprint,
        "source_lock_sha256": (
            sha256_file(source_lock_path) if source_lock_path.exists() else None
        ),
    }


def converted_state_manifest_sha256(converted: dict[str, Tensor]) -> str:
    manifest = [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(converted.items())
    ]
    return hashlib.sha256(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def render_messages(tokenizer: Any, messages: Sequence[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def tokenize_chat_records(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_len: int,
    limit: int | None = None,
) -> list[dict[str, Tensor]]:
    examples: list[dict[str, Tensor]] = []
    selected = records[:limit] if limit is not None else records
    for record in selected:
        raw_messages = record.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("record missing messages list")
        messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in raw_messages
        ]
        full_text = render_messages(tokenizer, messages)
        full_ids = _encode_ids(tokenizer, full_text)
        labels = [-100] * len(full_ids)
        for index, raw_message in enumerate(raw_messages):
            if raw_message.get("role") != "assistant" or raw_message.get("loss", True) is False:
                continue
            prefix_text = str(
                tokenizer.apply_chat_template(
                    messages[:index],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            through_text = render_messages(tokenizer, messages[: index + 1])
            prefix_ids = _encode_ids(tokenizer, prefix_text)
            through_ids = _encode_ids(tokenizer, through_text)
            if (
                full_ids[: len(prefix_ids)] != prefix_ids
                or full_ids[: len(through_ids)] != through_ids
                or len(through_ids) <= len(prefix_ids)
            ):
                raise ValueError(
                    "chat template is not prefix-stable; assistant-only loss cannot be proven"
                )
            labels[len(prefix_ids) : len(through_ids)] = full_ids[
                len(prefix_ids) : len(through_ids)
            ]
        input_values = full_ids[-max_seq_len:]
        label_values = labels[-max_seq_len:]
        if not any(value != -100 for value in label_values):
            raise ValueError("record has no assistant target inside max_seq_len")
        attention_values = [1] * len(input_values)
        padding = max_seq_len - len(input_values)
        input_values.extend([int(tokenizer.pad_token_id)] * padding)
        label_values.extend([-100] * padding)
        attention_values.extend([0] * padding)
        examples.append(
            {
                "input_ids": torch.tensor(input_values, dtype=torch.long),
                "attention_mask": torch.tensor(attention_values, dtype=torch.long),
                "labels": torch.tensor(label_values, dtype=torch.long),
            }
        )
    return examples


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"]
    if isinstance(values, Tensor):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def collate_batch(batch: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def collect_router_assignments(router_logits: Sequence[Tensor], top_k: int) -> dict[int, Tensor]:
    assignments: dict[int, Tensor] = {}
    for layer, logits in enumerate(router_logits):
        expert_ids, _ = topk_assignments(logits.detach(), top_k=top_k, normalize=True)
        assignments[layer] = expert_ids.cpu()
    return assignments


def save_checkpoint_metadata(
    output_dir: Path,
    *,
    step: int,
    config: WorkerConfig,
    trainable_counts: dict[str, int],
    expert_health: list[dict[str, Any]],
    conversion_manifest_sha256: str | None,
    accelerator: Any | None = None,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints" / f"step-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if accelerator is not None:
        accelerator.wait_for_everyone()
        accelerator.save_state(checkpoint_dir / "state")
    metadata = {
        "step": step,
        "created_at_unix": int(time.time()),
        "base_model": BANKING_V2_BASE_MODEL,
        "base_revision": BANKING_V2_BASE_REVISION,
        "generative_dataset": str(config.manifest),
        "dataset_identity": dataset_identity(config.manifest),
        "conversion_manifest_sha256": conversion_manifest_sha256,
        "ood_stock_response": BANKING_V2_OOD_STOCK_RESPONSE,
        "trainable_counts": trainable_counts,
        "expert_health": expert_health,
        "resume_validation": {
            "base_revision": BANKING_V2_BASE_REVISION,
            "manifest_path": str(config.manifest),
            "optimizer_scheduler_rng_state": accelerator is not None,
        },
    }
    path = checkpoint_dir / "metadata.json"
    if accelerator is None or accelerator.is_main_process:
        path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if accelerator is not None:
        accelerator.wait_for_everyone()
    return path


def run_training_loop(
    model: Any,
    tokenizer: Any,
    train_examples: Sequence[dict[str, Tensor]],
    validation_examples: Sequence[dict[str, Tensor]],
    config: WorkerConfig,
    *,
    use_accelerate: bool,
    conversion_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    trainable_counts = apply_banking_v2_trainable_policy(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.max_steps, 1)
    )
    loader = DataLoader(
        TokenizedChatDataset(train_examples),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    validation_loader = DataLoader(
        TokenizedChatDataset(validation_examples),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    accelerator = None
    if use_accelerate:
        from accelerate import (  # type: ignore[import-untyped]
            Accelerator,
            FullyShardedDataParallelPlugin,
        )

        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy="FULL_SHARD",
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["Qwen2MoeDecoderLayer"],
            state_dict_type="FULL_STATE_DICT",
            use_orig_params=True,
            sync_module_states=True,
            activation_checkpointing=True,
            limit_all_gathers=True,
        )
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
        accelerator = Accelerator(mixed_precision="bf16", fsdp_plugin=fsdp_plugin)
        model, optimizer, scheduler, loader, validation_loader = accelerator.prepare(
            model, optimizer, scheduler, loader, validation_loader
        )

    model.train()
    last_loss = None
    last_health: list[dict[str, Any]] = []
    step = 0
    if config.resume_from is not None:
        if accelerator is None:
            raise ValueError("resume requires the Accelerate remote path")
        metadata_path = config.resume_from / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("base_revision") != BANKING_V2_BASE_REVISION:
            raise ValueError("resume base revision does not match the pinned revision")
        if metadata.get("dataset_identity") != dataset_identity(config.manifest):
            raise ValueError("resume dataset identity does not match")
        if metadata.get("conversion_manifest_sha256") != conversion_manifest_sha256:
            raise ValueError("resume converted-state manifest does not match")
        state_path = config.resume_from / "state"
        required_state_groups = ("optimizer", "scheduler", "random_states")
        state_names = [path.name for path in state_path.iterdir()]
        missing_state = [
            group
            for group in required_state_groups
            if not any(group in name for name in state_names)
        ]
        if missing_state:
            raise ValueError(f"resume state is incomplete: {missing_state}")
        accelerator.load_state(state_path)
        step = int(metadata["step"])
    while step < config.max_steps:
        for batch in loader:
            step += 1
            outputs = model(**batch, output_router_logits=True)
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("model did not return a loss")
            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()
            grad_flags = routed_down_grad_flags(model)
            if accelerator is not None:
                accelerator.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=1.0,
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=1.0,
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            last_loss = float(loss.detach().cpu().item())

            router_logits = outputs.router_logits or ()
            if router_logits:
                model_config = (
                    accelerator.unwrap_model(model).config
                    if accelerator is not None
                    else model.config
                )
                assignments = collect_router_assignments(
                    router_logits, top_k=int(model_config.num_experts_per_tok)
                )
                health = expert_assignment_health(
                    assignments,
                    num_experts=int(model_config.num_experts),
                    aux_loss=outputs.aux_loss,
                    routed_down_grad_nonzero=grad_flags,
                )
                last_health = [asdict(item) for item in health]
                if step >= 250 and not expert_health_passed(health):
                    raise RuntimeError(
                        f"expert-health gate failed at optimizer step {step}; stopping run"
                    )

            if step % config.checkpoint_every == 0 or step == config.max_steps:
                save_checkpoint_metadata(
                    config.output_dir,
                    step=step,
                    config=config,
                    trainable_counts=trainable_counts,
                    expert_health=last_health,
                    conversion_manifest_sha256=conversion_manifest_sha256,
                    accelerator=accelerator,
                )
            if step >= config.max_steps:
                break

    model.eval()
    validation_loss_sum = 0.0
    validation_batches = 0
    with torch.inference_mode():
        for validation_batch in validation_loader:
            validation_outputs = model(**validation_batch, output_router_logits=False)
            if validation_outputs.loss is not None:
                value = validation_outputs.loss.detach()
                if accelerator is not None:
                    value = accelerator.gather_for_metrics(value.reshape(1)).mean()
                validation_loss_sum += float(value.cpu().item())
                validation_batches += 1

    final_dir = config.output_dir / "final"
    unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
    if accelerator is not None:
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        unwrapped.save_pretrained(
            final_dir,
            state_dict=state_dict,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            safe_serialization=True,
        )
        if accelerator.is_main_process and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(final_dir)
        accelerator.wait_for_everyone()
    else:
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(final_dir)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(final_dir)

    return {
        "steps": step,
        "last_loss": last_loss,
        "validation_loss": (
            validation_loss_sum / validation_batches if validation_batches else None
        ),
        "trainable_counts": trainable_counts,
        "expert_health_passed": all(item.get("passed", False) for item in last_health),
        "last_expert_health": last_health,
        "final_dir": str(final_dir),
    }


def tiny_smoke_records() -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a banking-services assistant."},
                {"role": "user", "content": "How do I replace my debit card?"},
                {"role": "assistant", "content": "I can help you replace your debit card."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a banking-services assistant."},
                {"role": "user", "content": "Tell me a football score."},
                {"role": "assistant", "content": BANKING_V2_OOD_STOCK_RESPONSE},
            ]
        },
    ]


def run_tiny_smoke(config: WorkerConfig) -> dict[str, Any]:
    from transformers.models.qwen2_moe import Qwen2MoeForCausalLM

    tokenizer = SimpleBankingTokenizer()
    model = Qwen2MoeForCausalLM(tiny_qwen2_moe_config())
    records = tiny_smoke_records()
    examples = tokenize_chat_records(records, tokenizer, max_seq_len=config.max_seq_len)
    return run_training_loop(
        model,
        tokenizer,
        examples,
        examples,
        config,
        use_accelerate=False,
        conversion_manifest_sha256=None,
    )


def run_remote_training(config: WorkerConfig) -> dict[str, Any]:
    assert_remote_execution_allowed(config)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2_moe import Qwen2MoeForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        BANKING_V2_BASE_MODEL, revision=BANKING_V2_BASE_REVISION
    )
    dense_model = AutoModelForCausalLM.from_pretrained(
        BANKING_V2_BASE_MODEL,
        revision=BANKING_V2_BASE_REVISION,
        torch_dtype=torch.bfloat16,
    )
    moe_config = banking_v2_qwen2_moe_config()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        moe_model = Qwen2MoeForCausalLM(moe_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    converted = convert_dense_qwen_to_banking_moe_state(dense_model.state_dict(), moe_config)
    conversion_hash = converted_state_manifest_sha256(converted)
    missing, unexpected = moe_model.load_state_dict(converted, strict=False)
    moe_model.tie_weights()
    del dense_model
    del converted
    allowed_missing = {"lm_head.weight"} if moe_config.tie_word_embeddings else set()
    disallowed_missing = set(missing) - allowed_missing
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "dense-to-MoE state mapping was incomplete: "
            f"missing={sorted(disallowed_missing)}, unexpected={sorted(unexpected)}"
        )
    if effective_parameter_count(moe_model) != BANKING_V2_TOTAL_PARAMETERS:
        raise RuntimeError(
            f"converted MoE parameter count {effective_parameter_count(moe_model):,} "
            f"does not match expected {BANKING_V2_TOTAL_PARAMETERS:,}"
        )

    train_records = load_manifest_records(config.manifest, "train")
    validation_records = load_manifest_records(config.manifest, "validation")
    train_examples = tokenize_chat_records(
        train_records, tokenizer, max_seq_len=config.max_seq_len
    )
    validation_examples = tokenize_chat_records(
        validation_records, tokenizer, max_seq_len=config.max_seq_len
    )
    result = run_training_loop(
        moe_model,
        tokenizer,
        train_examples,
        validation_examples,
        config,
        use_accelerate=True,
        conversion_manifest_sha256=conversion_hash,
    )
    result["load_state_missing"] = list(missing)
    result["load_state_unexpected"] = list(unexpected)
    if config.push_to_hub and int(os.environ.get("RANK", "0")) == 0:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=config.hub_dest, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=config.hub_dest,
            repo_type="model",
            folder_path=config.output_dir / "final",
        )
        result["pushed_to_hub"] = config.hub_dest
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = worker_config_from_args(args)
    if config.run_tiny_smoke:
        print(json.dumps(run_tiny_smoke(config), indent=2, sort_keys=True))
        return 0
    if config.dry_run and not remote_execution_allowed(config):
        print(json.dumps(build_dry_run_plan(config), indent=2, sort_keys=True))
        return 0
    print(json.dumps(run_remote_training(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
