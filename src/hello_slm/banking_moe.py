"""Banking-v2 Qwen2-MoE helpers.

This module is intentionally separate from the hello-world <500M trainer. It
defines the large banking-domain MoE experiment as a guarded, testable lane
without launching paid infrastructure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

BANKING_V2_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BANKING_V2_BASE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
BANKING_V2_HUB_DEST = "spkc83/hello-banking-moe-9b"
BANKING_V2_TOTAL_PARAMETERS = 8_943_713_792
BANKING_V2_ACTIVE_PARAMETERS = 2_073_443_840
BANKING_V2_GENERATIVE_DATASET = "data/banking-v2/manifest.json"
BANKING_V2_ROUTER_EVAL_DATASET = "PolyAI/banking77"
BANKING_V2_ROUTER_EVAL_REVISION = "90d4e2ee5521c04fc1488f065b8b083658768c57"
BANKING_V2_OOD_STOCK_RESPONSE = (
    "I can only help with retail banking and financial-services questions. "
    "Please ask about accounts, cards, transfers, payments, loans, or related banking support."
)
BANKING_V2_REQUIRED_DENSE_CONFIG_KEYS = frozenset(
    {
        "base_model",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "rope_theta",
        "tie_word_embeddings",
        "qkv_bias",
    }
)


@dataclass(frozen=True)
class BankingV2Pins:
    """Pinned package/model inputs for the cloud-intended banking-v2 run."""

    base_model: str = BANKING_V2_BASE_MODEL
    base_revision: str = BANKING_V2_BASE_REVISION
    generative_dataset: str = BANKING_V2_GENERATIVE_DATASET
    router_eval_dataset: str = BANKING_V2_ROUTER_EVAL_DATASET
    router_eval_revision: str = BANKING_V2_ROUTER_EVAL_REVISION
    transformers: str = ">=5.13,<5.14"
    accelerate: str = ">=1.12,<2"
    datasets: str = ">=4.4,<5"
    peft: str = ">=0.18,<0.19"
    trl: str = ">=0.26,<0.27"
    torch: str = ">=2.9,<3"


@dataclass(frozen=True)
class ExpertHealthThresholds:
    """Post-warmup routing health gate thresholds."""

    min_assignment_fraction: float = 0.005
    max_assignment_fraction: float = 0.20
    min_normalized_entropy: float = 0.75


def banking_v2_qwen2_moe_config() -> Any:
    """Return the exact Qwen2-MoE config for the banking-v2 9B lane."""

    from transformers.models.qwen2_moe import Qwen2MoeConfig

    return Qwen2MoeConfig(
        vocab_size=151_936,
        hidden_size=1_536,
        intermediate_size=8_960,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=32_768,
        tie_word_embeddings=True,
        qkv_bias=True,
        shared_expert_intermediate_size=8_960,
        moe_intermediate_size=2_048,
        num_experts=28,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        output_router_logits=True,
        router_aux_loss_coef=0.01,
        use_sliding_window=False,
        sliding_window=0,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
    )


def tiny_qwen2_moe_config(**overrides: Any) -> Any:
    """Return a small Qwen2-MoE config for local routing/backward smoke tests."""

    from transformers.models.qwen2_moe import Qwen2MoeConfig

    data: dict[str, Any] = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "tie_word_embeddings": True,
        "qkv_bias": True,
        "shared_expert_intermediate_size": 64,
        "moe_intermediate_size": 16,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "output_router_logits": True,
        "router_aux_loss_coef": 0.01,
        "use_sliding_window": False,
        "sliding_window": 0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
    }
    data.update(overrides)
    return Qwen2MoeConfig(**data)


def instantiate_banking_v2_meta_model(config: Any | None = None) -> Any:
    """Instantiate the banking-v2 model on the meta device."""

    from transformers.models.qwen2_moe import Qwen2MoeForCausalLM

    with torch.device("meta"):
        return Qwen2MoeForCausalLM(config or banking_v2_qwen2_moe_config())


def qwen2_moe_parameter_count(config: Any, *, tied_embeddings_once: bool = True) -> int:
    """Closed-form Qwen2-MoE parameter count using Transformers tensor shapes."""

    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    layers = int(config.num_hidden_layers)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = hidden // query_heads
    kv_hidden = kv_heads * head_dim
    shared_ff = int(config.shared_expert_intermediate_size)
    moe_ff = int(config.moe_intermediate_size)
    experts = int(config.num_experts)

    token_embedding = vocab * hidden
    lm_head = 0 if tied_embeddings_once and bool(config.tie_word_embeddings) else vocab * hidden
    attention = hidden * hidden + 2 * hidden * kv_hidden + hidden * hidden
    if bool(config.qkv_bias):
        attention += hidden + 2 * kv_hidden
    norms = 2 * hidden
    router = experts * hidden
    shared_expert = 3 * hidden * shared_ff + hidden
    routed_experts = experts * (3 * hidden * moe_ff)
    final_norm = hidden
    return token_embedding + lm_head + layers * (
        attention + norms + router + shared_expert + routed_experts
    ) + final_norm


def qwen2_moe_active_parameter_count(config: Any, *, tied_embeddings_once: bool = True) -> int:
    """Estimate parameters active per generated token: dense + shared + top-k routed experts."""

    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    layers = int(config.num_hidden_layers)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = hidden // query_heads
    kv_hidden = kv_heads * head_dim
    shared_ff = int(config.shared_expert_intermediate_size)
    moe_ff = int(config.moe_intermediate_size)
    top_k = int(config.num_experts_per_tok)
    experts = int(config.num_experts)

    token_embedding = vocab * hidden
    lm_head = 0 if tied_embeddings_once and bool(config.tie_word_embeddings) else vocab * hidden
    attention = hidden * hidden + 2 * hidden * kv_hidden + hidden * hidden
    if bool(config.qkv_bias):
        attention += hidden + 2 * kv_hidden
    norms = 2 * hidden
    router = experts * hidden
    shared_expert = 3 * hidden * shared_ff + hidden
    routed_active = top_k * (3 * hidden * moe_ff)
    final_norm = hidden
    return token_embedding + lm_head + layers * (
        attention + norms + router + shared_expert + routed_active
    ) + final_norm


def effective_parameter_count(model: Any) -> int:
    """Count parameters once under the banking-v2 tied embedding convention."""

    named_parameters = dict(model.named_parameters())
    total = sum(parameter.numel() for parameter in named_parameters.values())
    config = getattr(model, "config", None)
    if config is not None and bool(getattr(config, "tie_word_embeddings", False)):
        lm_head = getattr(model, "lm_head", None)
        if "lm_head.weight" in named_parameters and lm_head is not None and hasattr(
            lm_head, "weight"
        ):
            total -= int(lm_head.weight.numel())
    return total


def _layer_prefix(layer: int) -> str:
    return f"model.layers.{layer}"


def _dense_mlp_key(layer: int, projection: str) -> str:
    return f"{_layer_prefix(layer)}.mlp.{projection}_proj.weight"


def _moe_mlp_key(layer: int, suffix: str) -> str:
    return f"{_layer_prefix(layer)}.mlp.{suffix}"


def _take_rows(weight: Tensor, offset: int, rows: int) -> Tensor:
    indices = (torch.arange(rows, device=weight.device) + offset) % weight.shape[0]
    return weight.index_select(0, indices)


def _take_columns(weight: Tensor, offset: int, columns: int) -> Tensor:
    indices = (torch.arange(columns, device=weight.device) + offset) % weight.shape[1]
    return weight.index_select(1, indices)


def deterministic_noise(shape: torch.Size | tuple[int, ...], *, seed: int, std: float) -> Tensor:
    """Return deterministic small normal noise for conversion tests and jobs."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32) * std


def convert_dense_qwen_to_banking_moe_state(
    dense_state: Mapping[str, Tensor],
    config: Any,
    *,
    seed: int = 17,
    expert_noise_std: float = 1e-5,
    router_std: float = 1e-3,
) -> dict[str, Tensor]:
    """Create MoE-initialized tensors from dense Qwen MLP and non-MLP weights.

    The function is pure: callers provide a state mapping and receive converted
    tensors without mutating inputs or loading models.
    """

    converted: dict[str, Tensor] = {
        key: value.detach().clone() for key, value in dense_state.items() if ".mlp." not in key
    }
    if bool(config.tie_word_embeddings):
        converted.pop("lm_head.weight", None)
    layers = int(config.num_hidden_layers)
    experts = int(config.num_experts)
    moe_ff = int(config.moe_intermediate_size)
    hidden = int(config.hidden_size)

    for layer in range(layers):
        gate = dense_state[_dense_mlp_key(layer, "gate")]
        up = dense_state[_dense_mlp_key(layer, "up")]
        down = dense_state[_dense_mlp_key(layer, "down")]

        converted[_moe_mlp_key(layer, "shared_expert.gate_proj.weight")] = gate.detach().clone()
        converted[_moe_mlp_key(layer, "shared_expert.up_proj.weight")] = up.detach().clone()
        converted[_moe_mlp_key(layer, "shared_expert.down_proj.weight")] = (
            2.0 * down.detach().clone()
        )
        converted[_moe_mlp_key(layer, "shared_expert_gate.weight")] = torch.zeros(
            (1, hidden), dtype=gate.dtype, device=gate.device
        )

        gate_up = torch.empty(
            (experts, moe_ff * 2, hidden), dtype=gate.dtype, device=gate.device
        )
        routed_down = torch.zeros(
            (experts, hidden, moe_ff), dtype=down.dtype, device=down.device
        )
        for expert in range(experts):
            offset = (layer * 997 + expert * moe_ff) % gate.shape[0]
            expert_gate = _take_rows(gate, offset, moe_ff)
            expert_up = _take_rows(up, offset, moe_ff)
            noise_seed = seed + layer * 10_000 + expert
            noise = deterministic_noise(
                gate_up[expert].shape, seed=noise_seed, std=expert_noise_std
            ).to(dtype=gate.dtype, device=gate.device)
            gate_up[expert] = torch.cat([expert_gate, expert_up], dim=0) + noise
            # Requirement: routed down projections start as zero residual adapters.
        converted[_moe_mlp_key(layer, "experts.gate_up_proj")] = gate_up
        converted[_moe_mlp_key(layer, "experts.down_proj")] = routed_down
        converted[_moe_mlp_key(layer, "gate.weight")] = deterministic_noise(
            (experts, hidden), seed=seed + layer * 101, std=router_std
        ).to(dtype=gate.dtype, device=gate.device)

    return converted


def topk_assignments(
    router_logits: Tensor, *, top_k: int, normalize: bool = True
) -> tuple[Tensor, Tensor]:
    """Return top-k expert ids and routing probabilities from router logits."""

    probabilities = torch.softmax(router_logits.float(), dim=-1)
    weights, expert_ids = torch.topk(probabilities, k=top_k, dim=-1)
    if normalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return expert_ids, weights


@dataclass(frozen=True)
class LayerExpertHealth:
    layer: int
    min_assignment_fraction: float
    max_assignment_fraction: float
    normalized_entropy: float
    aux_loss_finite: bool
    routed_down_grad_nonzero: bool
    passed: bool


def expert_assignment_health(
    assignments_by_layer: Mapping[int, Tensor],
    *,
    num_experts: int,
    aux_loss: Tensor | float | None,
    routed_down_grad_nonzero: Mapping[int, bool],
    thresholds: ExpertHealthThresholds | None = None,
) -> list[LayerExpertHealth]:
    """Evaluate post-warmup expert utilization gates layer by layer."""

    thresholds = thresholds or ExpertHealthThresholds()
    if aux_loss is None:
        aux_finite = False
    elif isinstance(aux_loss, Tensor):
        aux_finite = bool(torch.isfinite(aux_loss.detach()).item())
    else:
        aux_finite = math.isfinite(float(aux_loss))

    results: list[LayerExpertHealth] = []
    for layer, assignments in sorted(assignments_by_layer.items()):
        flat = assignments.detach().reshape(-1).to(dtype=torch.long)
        counts = torch.bincount(flat.cpu(), minlength=num_experts).float()
        fractions = counts / counts.sum().clamp_min(1.0)
        nonzero = fractions[fractions > 0]
        entropy = (
            -(nonzero * nonzero.log()).sum() / math.log(num_experts)
            if len(nonzero)
            else torch.tensor(0.0)
        )
        min_fraction = float(fractions.min().item())
        max_fraction = float(fractions.max().item())
        entropy_value = float(entropy.item())
        grad_ok = bool(routed_down_grad_nonzero.get(layer, False))
        passed = (
            min_fraction >= thresholds.min_assignment_fraction
            and max_fraction <= thresholds.max_assignment_fraction
            and entropy_value >= thresholds.min_normalized_entropy
            and aux_finite
            and grad_ok
        )
        results.append(
            LayerExpertHealth(
                layer=layer,
                min_assignment_fraction=min_fraction,
                max_assignment_fraction=max_fraction,
                normalized_entropy=entropy_value,
                aux_loss_finite=aux_finite,
                routed_down_grad_nonzero=grad_ok,
                passed=passed,
            )
        )
    return results


def expert_health_passed(health: list[LayerExpertHealth]) -> bool:
    return bool(health) and all(item.passed for item in health)


def routed_down_grad_flags(model: Any) -> dict[int, bool]:
    """Return rolling nonzero-grad flags for routed expert down projections."""

    flags: dict[int, bool] = {}
    for name, parameter in model.named_parameters():
        marker = ".mlp.experts.down_proj"
        if marker not in name:
            continue
        parts = name.split(".")
        try:
            layer = int(parts[parts.index("layers") + 1])
        except (IndexError, ValueError):
            continue
        grad = parameter.grad.detach() if parameter.grad is not None else None
        flags[layer] = bool(
            grad is not None
            and torch.isfinite(grad).all().item()
            and grad.abs().sum().item() > 0
        )
    return flags


def validate_dense_config_details(dense_config_details: Mapping[str, Any]) -> None:
    """Validate that the fixed-prompt equivalence report captures dense details."""

    missing = sorted(BANKING_V2_REQUIRED_DENSE_CONFIG_KEYS - set(dense_config_details))
    if missing:
        raise ValueError(f"dense_config_details missing required keys: {', '.join(missing)}")


def dense_logit_equivalence_report(
    dense_logits: Tensor,
    converted_logits: Tensor,
    *,
    atol: float = 1e-5,
    dense_config_details: Mapping[str, Any],
) -> dict[str, Any]:
    """Report the pre-update fixed-prompt dense-vs-converted logit check."""

    validate_dense_config_details(dense_config_details)
    max_abs_diff = float((dense_logits.float() - converted_logits.float()).abs().max().item())
    return {
        "passed": max_abs_diff <= atol,
        "max_abs_diff": max_abs_diff,
        "atol": atol,
        "mode": "fp32_eval_fixed_prompt_pre_update",
        "dense_config": dict(dense_config_details),
    }


def banking_v2_trainable_policy(parameter_name: str) -> str:
    """Classify a banking-v2 parameter name as trainable, LoRA-trainable, or frozen."""

    if "lora_" in parameter_name:
        return "train_lora_adapter"
    if parameter_name.endswith(".mlp.gate.weight"):
        return "train_router"
    if ".mlp.experts.down_proj" in parameter_name:
        return "train_routed_residual_adapter"
    return "freeze_copied_base"


def banking_v2_trainable_names(named_parameters: Mapping[str, Tensor]) -> list[str]:
    """Return trainable parameter names under the banking-v2 adaptation policy."""

    return [
        name
        for name in named_parameters
        if banking_v2_trainable_policy(name) != "freeze_copied_base"
    ]


def apply_banking_v2_trainable_policy(model: Any) -> dict[str, int]:
    """Apply the banking-v2 train/freeze policy to a model in place."""

    counts = {"trainable": 0, "frozen": 0}
    for name, parameter in model.named_parameters():
        trainable = banking_v2_trainable_policy(name) != "freeze_copied_base"
        parameter.requires_grad = trainable
        key = "trainable" if trainable else "frozen"
        counts[key] += int(parameter.numel())
    return counts


def banking_v2_training_summary(config: Any | None = None) -> dict[str, Any]:
    """Summarize the cloud-intended training plan without side effects."""

    cfg = config or banking_v2_qwen2_moe_config()
    total = qwen2_moe_parameter_count(cfg)
    active = qwen2_moe_active_parameter_count(cfg)
    hidden = int(cfg.hidden_size)
    layers = int(cfg.num_hidden_layers)
    experts = int(cfg.num_experts)
    moe_ff = int(cfg.moe_intermediate_size)
    routed_down_trainable = layers * experts * hidden * moe_ff
    router_trainable = layers * experts * hidden
    trainable = routed_down_trainable + router_trainable
    frozen = total - trainable
    return {
        "base_model": BANKING_V2_BASE_MODEL,
        "base_revision": BANKING_V2_BASE_REVISION,
        "hub_dest": BANKING_V2_HUB_DEST,
        "job_status": "executable_worker_implemented_not_launched",
        "scope": "sft_domain_adaptation_not_from_scratch_pretraining",
        "generative_sft_dataset": BANKING_V2_GENERATIVE_DATASET,
        "router_eval_dataset": BANKING_V2_ROUTER_EVAL_DATASET,
        "out_of_domain_stock_response": BANKING_V2_OOD_STOCK_RESPONSE,
        "total_parameters": total,
        "estimated_active_parameters": active,
        "estimated_trainable_parameters": trainable,
        "estimated_routed_down_trainable_parameters": routed_down_trainable,
        "estimated_router_trainable_parameters": router_trainable,
        "estimated_mostly_frozen_parameters": frozen,
        "trainable_rules": [
            "model.layers.*.mlp.gate.weight",
            "model.layers.*.mlp.experts.down_proj",
        ],
        "frozen_rules": [
            "model.embed_tokens.weight and tied lm_head.weight",
            "copied dense self-attention and layernorm weights",
            "copied shared expert weights",
            "initialized routed expert gate/up projections",
        ],
        "hardware": "A100x4",
        "fsdp": "full_shard",
        "precision": "bf16",
        "activation_checkpointing": True,
        "checkpoint_every_steps": 250,
        "max_wallclock_hours": 10,
        "max_budget_usd": 100,
    }
