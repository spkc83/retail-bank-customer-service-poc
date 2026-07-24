"""Decoder-only Transformer model for the hello-SLM pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PARAMETER_CAP = 500_000_000


@dataclass(frozen=True)
class ModelConfig:
    """Spec-version-1 decoder-only Transformer configuration."""

    vocab_size: int
    max_seq_len: int
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    pad_token_id: int = 0
    norm: str = "rmsnorm"
    mlp: str = "swiglu"
    rope_base: float = 10_000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    init_std: float = 0.02
    residual_init_scale: str = "deepnorm-lite"
    parameter_cap: int = PARAMETER_CAP
    architecture: str = "decoder_only_causal_transformer"
    format_version: int = 1

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ModelConfig:
        return cls(**{field: data[field] for field in cls.__dataclass_fields__ if field in data})

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError("model.format_version must be 1")
        if self.architecture != "decoder_only_causal_transformer":
            raise ValueError("architecture must be decoder_only_causal_transformer")
        positive_fields = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_ff": self.d_ff,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be inside the vocabulary")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.n_kv_heads != self.n_heads:
            raise ValueError("spec version 1 supports full multi-head attention only")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even per-head dimension")
        if self.norm != "rmsnorm":
            raise ValueError("norm must be rmsnorm")
        if self.mlp != "swiglu":
            raise ValueError("mlp must be swiglu")
        if not self.tie_embeddings:
            raise ValueError("tie_embeddings must be true")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.init_std <= 0.0:
            raise ValueError("init_std must be positive")
        if self.rope_base <= 0.0:
            raise ValueError("rope_base must be positive")
        if self.residual_init_scale not in {"deepnorm-lite", "none"}:
            raise ValueError("residual_init_scale must be deepnorm-lite or none")
        assert_parameter_cap(parameter_count_from_config(self), self.parameter_cap)


def parameter_count_from_config(config: ModelConfig | dict[str, Any]) -> int:
    """Compute the exact tied-embedding parameter count without allocating a model."""

    cfg = config if isinstance(config, ModelConfig) else ModelConfig.from_mapping(config)
    embedding = cfg.vocab_size * cfg.d_model
    per_layer_attention = 4 * cfg.d_model * cfg.d_model
    per_layer_mlp = 3 * cfg.d_model * cfg.d_ff
    per_layer_norms = 2 * cfg.d_model
    final_norm = cfg.d_model
    return embedding + cfg.n_layers * (
        per_layer_attention + per_layer_mlp + per_layer_norms
    ) + final_norm


def assert_parameter_cap(parameter_count: int, cap: int = PARAMETER_CAP) -> None:
    if parameter_count >= cap:
        raise ValueError(f"parameter count {parameter_count:,} must be below {cap:,}")


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float) -> None:
        super().__init__()
        self.cos: Tensor
        self.sin: Tensor
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        pair_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = base ** (-pair_indices / head_dim)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        seq_len = q.size(1)
        cos = self.cos[:seq_len].to(dtype=q.dtype, device=q.device)[None, :, None, :]
        sin = self.sin[:seq_len].to(dtype=q.dtype, device=q.device)[None, :, None, :]
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_base)

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q, k = self.rope(q, k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask[:, None, :, :],
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask)
        return x + self.mlp(self.mlp_norm(x))


class HelloSLMModel(nn.Module):
    def __init__(self, config: ModelConfig | dict[str, Any]) -> None:
        super().__init__()
        self.config = (
            config if isinstance(config, ModelConfig) else ModelConfig.from_mapping(config)
        )
        self.token_embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.d_model,
            padding_idx=self.config.pad_token_id,
        )
        self.blocks = nn.ModuleList(DecoderBlock(self.config) for _ in range(self.config.n_layers))
        self.final_norm = RMSNorm(self.config.d_model)
        self._residual_projections = {
            projection
            for block in self.blocks
            if isinstance(block, DecoderBlock)
            for projection in (block.attn.out_proj, block.mlp.down_proj)
        }
        self.apply(self._init_weights)

    @property
    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def forward(
        self,
        token_ids: Tensor,
        assistant_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, seq]")
        if token_ids.size(1) > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.config.vocab_size:
            raise ValueError("token_ids contain IDs outside the vocabulary")

        attention_mask = self._attention_mask(token_ids)
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.final_norm(x)
        logits = x @ self.token_embedding.weight.T
        loss = None
        if assistant_mask is not None:
            loss = self.loss(logits, token_ids, assistant_mask)
        return logits, loss

    def loss(self, logits: Tensor, token_ids: Tensor, assistant_mask: Tensor) -> Tensor:
        if assistant_mask.shape != token_ids.shape:
            raise ValueError("assistant_mask must match token_ids shape")
        labels = token_ids[:, 1:]
        predicted = logits[:, :-1, :]
        loss_mask = assistant_mask[:, 1:].bool() & labels.ne(self.config.pad_token_id)
        per_token = F.cross_entropy(
            predicted.reshape(-1, self.config.vocab_size),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)
        denominator = loss_mask.sum().clamp_min(1)
        return (per_token * loss_mask).sum() / denominator

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        stop_ids: set[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if prompt_ids.ndim == 1:
            output = prompt_ids.unsqueeze(0).clone()
        elif prompt_ids.ndim == 2:
            output = prompt_ids.clone()
        else:
            raise ValueError("prompt_ids must have shape [seq] or [batch, seq]")
        if output.size(0) != 1:
            raise ValueError("generate currently supports batch size 1")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive")
        stop_ids = stop_ids or set()

        for _ in range(max_new_tokens):
            if output.size(1) >= self.config.max_seq_len:
                break
            logits, _ = self(output)
            next_logits = logits[:, -1, :].clone()
            if repetition_penalty != 1.0:
                seen = output[0].unique()
                next_logits[:, seen] = torch.where(
                    next_logits[:, seen] < 0,
                    next_logits[:, seen] * repetition_penalty,
                    next_logits[:, seen] / repetition_penalty,
                )
            if temperature == 0.0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                next_logits = _filter_logits(next_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            output = torch.cat([output, next_id], dim=1)
            if int(next_id.item()) in stop_ids:
                break
        return output

    def _attention_mask(self, token_ids: Tensor) -> Tensor:
        seq_len = token_ids.size(1)
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=token_ids.device).tril()
        key_is_not_pad = token_ids.ne(self.config.pad_token_id)[:, None, :]
        return causal[None, :, :] & key_is_not_pad

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = self.config.init_std
            if (
                self.config.residual_init_scale == "deepnorm-lite"
                and module in self._residual_projections
            ):
                std = self.config.init_std / math.sqrt(2 * self.config.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)


def _filter_logits(logits: Tensor, *, top_k: int | None, top_p: float | None) -> Tensor:
    filtered = logits
    if top_k is not None and top_k < filtered.size(-1):
        threshold = torch.topk(filtered, top_k, dim=-1).values[:, -1, None]
        filtered = filtered.masked_fill(filtered < threshold, torch.finfo(filtered.dtype).min)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(sorted_logits.dtype).min)
        filtered = torch.full_like(filtered, torch.finfo(filtered.dtype).min)
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered
