from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hello_slm.model import (  # noqa: E402
    CausalSelfAttention,
    HelloSLMModel,
    ModelConfig,
    assert_parameter_cap,
    parameter_count_from_config,
)


def smoke_config() -> ModelConfig:
    with (ROOT / "configs" / "smoke.toml").open("rb") as handle:
        data = tomllib.load(handle)
    return ModelConfig.from_mapping(data["model"])


def test_parameter_count_matches_smoke_config_and_allocated_model() -> None:
    torch.manual_seed(104)
    config = smoke_config()
    model = HelloSLMModel(config)

    assert parameter_count_from_config(config) == 123_200
    assert model.parameter_count == 123_200
    assert model.parameter_count == parameter_count_from_config(config)


def test_parameter_cap_boundary() -> None:
    assert_parameter_cap(499_999_999)
    with pytest.raises(ValueError, match="below 500,000,000"):
        assert_parameter_cap(500_000_000)


def test_parameter_formula_uses_tied_embedding_transformer_terms() -> None:
    config = ModelConfig(
        vocab_size=17,
        max_seq_len=8,
        n_layers=3,
        d_model=12,
        n_heads=3,
        n_kv_heads=3,
        d_ff=20,
    )

    expected = 17 * 12
    expected += 3 * ((4 * 12 * 12) + (3 * 12 * 20) + (2 * 12))
    expected += 12
    assert parameter_count_from_config(config) == expected


def test_config_invariants_reject_unsupported_shapes_and_layouts() -> None:
    base = smoke_config().__dict__

    with pytest.raises(ValueError, match="d_model must be divisible"):
        ModelConfig.from_mapping({**base, "d_model": 66})
    with pytest.raises(ValueError, match="full multi-head attention"):
        ModelConfig.from_mapping({**base, "n_kv_heads": 2})
    with pytest.raises(ValueError, match="tie_embeddings"):
        ModelConfig.from_mapping({**base, "tie_embeddings": False})
    with pytest.raises(ValueError, match="norm"):
        ModelConfig.from_mapping({**base, "norm": "layernorm"})
    with pytest.raises(ValueError, match="mlp"):
        ModelConfig.from_mapping({**base, "mlp": "gelu"})


def test_forward_shapes_finite_assistant_loss_and_padding() -> None:
    torch.manual_seed(104)
    model = HelloSLMModel(smoke_config())
    token_ids = torch.tensor(
        [
            [2, 5, 6, 7, 3, 0, 0],
            [2, 8, 9, 3, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    assistant_mask = torch.tensor(
        [
            [False, False, True, True, True, False, False],
            [False, True, True, True, False, False, False],
        ]
    )

    logits, loss = model(token_ids, assistant_mask=assistant_mask)

    assert logits.shape == (2, 7, 256)
    assert loss is not None
    assert torch.isfinite(loss)


def test_attention_mask_is_causal_and_excludes_padding_keys() -> None:
    model = HelloSLMModel(smoke_config())
    token_ids = torch.tensor([[2, 5, 0, 0]], dtype=torch.long)

    mask = model._attention_mask(token_ids)

    assert mask.tolist() == [
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, False, False],
            [True, True, False, False],
        ]
    ]


def _manual_attention_reference(
    attention: CausalSelfAttention,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, d_model = x.shape
    q = attention.q_proj(x).view(batch_size, seq_len, attention.n_heads, attention.head_dim)
    k = attention.k_proj(x).view(batch_size, seq_len, attention.n_heads, attention.head_dim)
    v = attention.v_proj(x).view(batch_size, seq_len, attention.n_heads, attention.head_dim)
    q, k = attention.rope(q, k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    expanded_mask = mask[:, None, :, :]
    scores = q @ k.transpose(-2, -1) / (attention.head_dim**0.5)
    scores = scores.masked_fill(~expanded_mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    weights = weights.masked_fill(~expanded_mask, 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    y = weights @ v
    y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    return attention.out_proj(y)


@pytest.mark.parametrize(
    "token_ids",
    [
        torch.tensor([[2, 5, 0, 0]], dtype=torch.long),
        torch.tensor([[0, 0, 2, 5]], dtype=torch.long),
        torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
    ],
)
def test_sdpa_attention_matches_manual_mask_reference(token_ids: torch.Tensor) -> None:
    torch.manual_seed(104)
    model = HelloSLMModel(smoke_config())
    attention = model.blocks[0].attn
    assert isinstance(attention, CausalSelfAttention)
    attention.eval()
    x = torch.randn(token_ids.size(0), token_ids.size(1), model.config.d_model)
    mask = model._attention_mask(token_ids)

    actual = attention(x, mask)
    expected = _manual_attention_reference(attention, x, mask)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_causal_attention_prevents_future_tokens_from_affecting_past_logits() -> None:
    torch.manual_seed(104)
    model = HelloSLMModel(smoke_config())
    left = torch.tensor([[2, 5, 6, 7, 3, 0, 0]], dtype=torch.long)
    changed_future = torch.tensor([[2, 5, 6, 7, 99, 88, 77]], dtype=torch.long)

    left_logits, _ = model(left)
    changed_logits, _ = model(changed_future)

    torch.testing.assert_close(
        left_logits[:, :4, :],
        changed_logits[:, :4, :],
        atol=1e-5,
        rtol=1e-5,
    )


def test_generation_is_bounded_vocab_safe_and_can_stop() -> None:
    torch.manual_seed(104)
    model = HelloSLMModel(smoke_config())
    prompt = torch.tensor([2, 5, 6], dtype=torch.long)
    generator = torch.Generator().manual_seed(106)

    generated = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.05,
        stop_ids={3},
        generator=generator,
    )

    assert generated.ndim == 2
    assert generated.shape[0] == 1
    assert generated.shape[1] <= prompt.numel() + 5
    assert int(generated.min()) >= 0
    assert int(generated.max()) < smoke_config().vocab_size


def test_generation_stops_at_context_cap() -> None:
    torch.manual_seed(104)
    model = HelloSLMModel(smoke_config())
    prompt = torch.tensor([2] * (model.config.max_seq_len - 1), dtype=torch.long)

    generated = model.generate(
        prompt,
        max_new_tokens=16,
        generator=torch.Generator().manual_seed(106),
    )

    assert generated.shape == (1, model.config.max_seq_len)


def test_zero_temperature_generation_is_deterministic_greedy() -> None:
    torch.manual_seed(104)
    model = HelloSLMModel(smoke_config())
    prompt = torch.tensor([2, 5, 6], dtype=torch.long)

    first = model.generate(prompt, max_new_tokens=3, temperature=0.0)
    second = model.generate(prompt, max_new_tokens=3, temperature=0.0)

    torch.testing.assert_close(first, second)
