from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import tomllib

import pytest
import torch

from hello_slm.banking_moe import (
    BANKING_V2_ACTIVE_PARAMETERS,
    BANKING_V2_BASE_REVISION,
    BANKING_V2_OOD_STOCK_RESPONSE,
    BANKING_V2_TOTAL_PARAMETERS,
    banking_v2_qwen2_moe_config,
    banking_v2_trainable_names,
    banking_v2_trainable_policy,
    convert_dense_qwen_to_banking_moe_state,
    dense_logit_equivalence_report,
    effective_parameter_count,
    expert_assignment_health,
    expert_health_passed,
    instantiate_banking_v2_meta_model,
    qwen2_moe_active_parameter_count,
    qwen2_moe_parameter_count,
    routed_down_grad_flags,
    tiny_qwen2_moe_config,
    topk_assignments,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="banking-v2 MoE tests require transformers",
)


def test_banking_v2_exact_qwen2_moe_config_and_meta_parameter_count() -> None:
    cfg = banking_v2_qwen2_moe_config()

    assert cfg.vocab_size == 151_936
    assert cfg.hidden_size == 1_536
    assert cfg.num_hidden_layers == 28
    assert cfg.num_attention_heads == 12
    assert cfg.num_key_value_heads == 2
    assert cfg.rope_parameters["rope_theta"] == 1_000_000.0
    assert cfg.use_sliding_window is False
    assert cfg.tie_word_embeddings is True
    assert cfg.qkv_bias is True
    assert cfg.shared_expert_intermediate_size == 8_960
    assert cfg.moe_intermediate_size == 2_048
    assert cfg.num_experts == 28
    assert cfg.num_experts_per_tok == 2
    assert cfg.norm_topk_prob is True
    assert cfg.output_router_logits is True
    assert cfg.router_aux_loss_coef == 0.01

    model = instantiate_banking_v2_meta_model(cfg)
    assert effective_parameter_count(model) == BANKING_V2_TOTAL_PARAMETERS
    assert qwen2_moe_parameter_count(cfg) == BANKING_V2_TOTAL_PARAMETERS
    assert qwen2_moe_active_parameter_count(cfg) == BANKING_V2_ACTIVE_PARAMETERS
    assert qwen2_moe_active_parameter_count(cfg) / 1_000_000_000 == pytest.approx(2.073, abs=0.001)


def test_dense_to_moe_conversion_policy_is_deterministic_and_pure() -> None:
    cfg = tiny_qwen2_moe_config(num_hidden_layers=1, hidden_size=8, intermediate_size=16)
    gate_weight = torch.arange(16 * 8, dtype=torch.float32).view(16, 8)
    up_weight = torch.arange(200, 200 + 16 * 8, dtype=torch.float32).view(16, 8)
    down_weight = torch.arange(400, 400 + 8 * 16, dtype=torch.float32).view(8, 16)
    dense_state = {
        "model.embed_tokens.weight": torch.arange(128 * 8, dtype=torch.float32).view(128, 8),
        "lm_head.weight": torch.full((128, 8), 99.0),
        "model.layers.0.self_attn.q_proj.weight": torch.ones(8, 8),
        "model.layers.0.mlp.gate_proj.weight": gate_weight,
        "model.layers.0.mlp.up_proj.weight": up_weight,
        "model.layers.0.mlp.down_proj.weight": down_weight,
    }

    first = convert_dense_qwen_to_banking_moe_state(dense_state, cfg, seed=7)
    second = convert_dense_qwen_to_banking_moe_state(dense_state, cfg, seed=7)
    zero_noise = convert_dense_qwen_to_banking_moe_state(
        dense_state, cfg, seed=7, expert_noise_std=0.0
    )

    assert first["model.layers.0.self_attn.q_proj.weight"].data_ptr() != dense_state[
        "model.layers.0.self_attn.q_proj.weight"
    ].data_ptr()
    assert torch.equal(first["model.layers.0.self_attn.q_proj.weight"], torch.ones(8, 8))
    assert "lm_head.weight" not in first
    assert torch.equal(
        first["model.layers.0.mlp.shared_expert.gate_proj.weight"],
        dense_state["model.layers.0.mlp.gate_proj.weight"],
    )
    assert torch.equal(
        first["model.layers.0.mlp.shared_expert.up_proj.weight"],
        dense_state["model.layers.0.mlp.up_proj.weight"],
    )
    assert torch.equal(
        first["model.layers.0.mlp.shared_expert.down_proj.weight"],
        2.0 * dense_state["model.layers.0.mlp.down_proj.weight"],
    )
    assert torch.count_nonzero(first["model.layers.0.mlp.shared_expert_gate.weight"]) == 0
    assert torch.count_nonzero(first["model.layers.0.mlp.experts.down_proj"]) == 0
    assert torch.count_nonzero(first["model.layers.0.mlp.gate.weight"]) > 0
    assert torch.equal(
        first["model.layers.0.mlp.experts.gate_up_proj"],
        second["model.layers.0.mlp.experts.gate_up_proj"],
    )
    assert torch.equal(
        first["model.layers.0.mlp.gate.weight"],
        second["model.layers.0.mlp.gate.weight"],
    )
    assert torch.equal(
        zero_noise["model.layers.0.mlp.experts.gate_up_proj"][0, :16],
        gate_weight,
    )
    assert torch.equal(
        zero_noise["model.layers.0.mlp.experts.gate_up_proj"][0, 16:],
        up_weight,
    )


def test_expert_health_gate_accepts_balanced_post_warmup_assignments() -> None:
    assignments = torch.arange(28).repeat(8).view(28, 8)
    health = expert_assignment_health(
        {0: assignments},
        num_experts=28,
        aux_loss=torch.tensor(0.01),
        routed_down_grad_nonzero={0: True},
    )

    assert len(health) == 1
    assert health[0].min_assignment_fraction >= 0.005
    assert health[0].max_assignment_fraction <= 0.20
    assert health[0].normalized_entropy == pytest.approx(1.0)
    assert expert_health_passed(health)


def test_expert_health_gate_rejects_collapsed_or_missing_gradients() -> None:
    collapsed = torch.zeros((20, 2), dtype=torch.long)
    health = expert_assignment_health(
        {0: collapsed},
        num_experts=28,
        aux_loss=torch.tensor(float("nan")),
        routed_down_grad_nonzero={0: False},
    )

    assert not health[0].passed
    assert health[0].max_assignment_fraction == 1.0
    assert health[0].aux_loss_finite is False
    assert not expert_health_passed(health)


def test_tiny_moe_forward_backward_exercises_routing_and_gradients() -> None:
    from transformers.models.qwen2_moe import Qwen2MoeForCausalLM

    torch.manual_seed(1234)
    cfg = tiny_qwen2_moe_config()
    model = Qwen2MoeForCausalLM(cfg)
    token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    outputs = model(input_ids=token_ids, labels=token_ids, output_router_logits=True)

    assert outputs.loss is not None
    assert outputs.aux_loss is not None
    assert outputs.router_logits is not None
    assert len(outputs.router_logits) == cfg.num_hidden_layers
    expert_ids, weights = topk_assignments(
        outputs.router_logits[0], top_k=cfg.num_experts_per_tok, normalize=True
    )
    assert expert_ids.shape[-1] == cfg.num_experts_per_tok
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))

    outputs.loss.backward()
    flags = routed_down_grad_flags(model)
    assert set(flags) == {0, 1}
    assert all(flags.values())
    assert math.isfinite(float(outputs.aux_loss.detach()))

    for name, parameter in model.named_parameters():
        if ".layers.0.mlp.experts.down_proj" in name and parameter.grad is not None:
            parameter.grad.reshape(-1)[0] = float("nan")
            break
    assert routed_down_grad_flags(model)[0] is False


def test_dense_logit_equivalence_report_carries_dense_details() -> None:
    dense_logits = torch.tensor([[[1.0, 2.0]]])
    converted_logits = dense_logits + 5e-6
    report = dense_logit_equivalence_report(
        dense_logits,
        converted_logits,
        dense_config_details={
            "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "vocab_size": 151_936,
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "intermediate_size": 8960,
            "rope_theta": 1_000_000.0,
            "tie_word_embeddings": True,
            "qkv_bias": True,
        },
    )

    assert report["passed"] is True
    assert report["max_abs_diff"] <= 1e-5
    assert report["mode"] == "fp32_eval_fixed_prompt_pre_update"
    assert report["dense_config"]["hidden_size"] == 1536


def test_dense_logit_equivalence_report_requires_full_dense_config() -> None:
    with pytest.raises(ValueError, match="dense_config_details missing"):
        dense_logit_equivalence_report(
            torch.tensor([[[1.0]]]),
            torch.tensor([[[1.0]]]),
            dense_config_details={"base_model": "Qwen/Qwen2.5-1.5B-Instruct"},
        )


def test_banking_v2_trainable_policy_is_executable() -> None:
    named = {
        "model.embed_tokens.weight": torch.zeros(1),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        "model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.zeros(1),
        "model.layers.0.mlp.gate.weight": torch.zeros(1),
        "model.layers.0.mlp.experts.gate_up_proj": torch.zeros(1),
        "model.layers.0.mlp.experts.down_proj": torch.zeros(1),
        "model.layers.0.mlp.shared_expert.gate_proj.weight": torch.zeros(1),
    }

    assert banking_v2_trainable_policy("model.embed_tokens.weight") == "freeze_copied_base"
    assert banking_v2_trainable_policy("model.layers.0.mlp.gate.weight") == "train_router"
    assert (
        banking_v2_trainable_policy("model.layers.0.mlp.experts.down_proj")
        == "train_routed_residual_adapter"
    )
    assert (
        banking_v2_trainable_policy("model.layers.0.self_attn.q_proj.lora_A.default.weight")
        == "train_lora_adapter"
    )
    assert banking_v2_trainable_names(named) == [
        "model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.experts.down_proj",
    ]


def test_banking_v2_job_script_is_dry_run_guarded() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/banking_v2/train_banking_moe.py"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"hub_dest": "spkc83/retail-bank-servicing-moe-9b"' in result.stdout
    assert '"base_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"' in result.stdout
    assert '"total_parameters": 8943713792' in result.stdout
    assert '"estimated_trainable_parameters"' in result.stdout
    assert '"resume_validation"' in result.stdout
    assert '"generative_sft_dataset": "data/banking-v2/manifest.json"' in result.stdout
    assert '"router_eval_dataset": "PolyAI/banking77"' in result.stdout
    assert '"rtx-pro-6000"' in result.stdout
    assert "executable_worker_implemented_not_launched" in result.stdout
    assert "Dry-run only" in result.stdout


def test_banking_v2_configs_pin_revision_dataset_and_ood_response() -> None:
    for config_path in [
        "configs/banking-v2-dense-adapter.toml",
        "configs/banking-v2-moe-9b.toml",
    ]:
        with open(config_path, "rb") as handle:
            config = tomllib.load(handle)

        assert config["model"]["base_revision"] == BANKING_V2_BASE_REVISION
        assert config["dataset"]["name"] == "data/banking-v2/manifest.json"
        assert config["dataset"]["source"] == "prepared_banking_v2_bitext_composition"
        assert config["dataset"]["router_eval_name"] == "PolyAI/banking77"
        assert config["dataset"]["out_of_domain_stock_response"] == BANKING_V2_OOD_STOCK_RESPONSE
