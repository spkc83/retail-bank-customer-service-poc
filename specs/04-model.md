# Model specification

This document is normative. The model is a decoder-only causal Transformer
initialized from random weights. Importing pretrained weights, embeddings,
tokenizers, adapters, or optimizer state is forbidden.

## Architecture

The reference model is a GPT-style stack:

```text
token_ids[B, T]
  -> token_embedding[V, D]
  -> N x decoder_block
  -> final_rms_norm[D]
  -> logits[B, T, V] using tied token_embedding transpose
```

All tensors use batch-major layout unless stated otherwise. `B` is batch size,
`T` is sequence length, `V` is vocabulary size, `D` is model width, `N` is layer
count, `H` is query attention head count, `K` is key/value head count, `Dh = D/H`,
and `F` is SwiGLU hidden width. `D MUST be divisible by H`; `H MUST be divisible
by K`. The hello-world profiles use full multi-head attention where `K = H`.
Grouped-query attention MAY be added later only if the parameter estimator is
versioned with the implementation.

Each decoder block is pre-norm:

```text
x[B,T,D]
  a = rms_norm_attn(x)
  q = a @ Wq[D,D]      -> [B,T,H,Dh]
  k = a @ Wk[D,D]      -> [B,T,H,Dh]
  v = a @ Wv[D,D]      -> [B,T,H,Dh]
  q,k = apply_rope(q,k)
  y = causal_attention(q,k,v,mask) -> [B,T,D]
  x = x + (y @ Wo[D,D])
  m = rms_norm_mlp(x)
  x = x + (silu(m @ Wgate[D,F]) * (m @ Wup[D,F])) @ Wdown[F,D]
```

Linear layers have no bias. RMSNorm has one learned scale vector and no bias.
Dropout defaults to zero in both checked-in profiles so deterministic smoke tests
can compare checkpoints exactly. A nonzero dropout value is allowed only when its
RNG state is persisted in checkpoints.

## Token and position handling

The tokenizer defines immutable IDs for at least `<pad>`, `<unk>`, `<bos>`,
`<eos>`, `<user>`, `<assistant>`, and `<system>`. The model does not encode role
semantics beyond those token IDs.

The model has no learned absolute position embedding. Position information is
Rotary Position Embedding (RoPE) applied to query and key tensors before the
attention dot product. RoPE is applied over the full per-head dimension `Dh`.
For token position `p` and even channel pair `i`, use:

```text
theta_i = rope_base ^ (-2i / Dh)
rotate(x_2i, x_2i+1, p) =
  (x_2i * cos(p * theta_i) - x_2i+1 * sin(p * theta_i),
   x_2i * sin(p * theta_i) + x_2i+1 * cos(p * theta_i))
```

`rope_base = 10000` for the checked-in profiles. Training and serving MUST reject
prompts longer than `max_seq_len`; extrapolation or RoPE scaling is out of scope.

## Attention mask and loss alignment

The attention mask is the logical AND of:

- a lower-triangular causal mask where token `t` can attend only to positions
  `<= t`;
- a padding mask where `<pad>` positions cannot be attended to by non-pad tokens.

Training predicts `token_ids[:, 1:]` from logits at `[:, :-1]`. Loss MUST ignore
`<pad>` labels and MUST mask user/system prompt spans when the packed example
marks them as non-assistant. Assistant tokens, including assistant `<eos>`, are
loss-bearing.

## Initialization

All model weights are initialized from the configured seed after tokenizer and
dataset construction have completed. Initialization MUST be reproducible for a
fixed effective config and device class.

- `token_embedding.weight[V,D]`: normal mean `0`, std `init_std`.
- `Wq`, `Wk`, `Wv`, `Wo`, `Wgate`, `Wup`, `Wdown`: normal mean `0`, std
  `init_std`.
- Residual projection scaling: `Wo` and `Wdown` MAY use std
  `init_std / sqrt(2N)` when `residual_init_scale = "deepnorm-lite"`; checked-in
  profiles use this setting.
- RMSNorm scales initialize to `1`.

## Exact parameter formula

The validator MUST compute exact model parameters from config without allocating
the target model. With full multi-head attention, tied embeddings, no linear
biases, no learned position embeddings, and RMSNorm, the count is:

```text
embedding = V * D
per_layer_attention = 4 * D * D
per_layer_mlp = 3 * D * F
per_layer_norms = 2 * D
final_norm = D
total = embedding + N * (per_layer_attention + per_layer_mlp + per_layer_norms) + final_norm
```

The tied output projection contributes no additional parameters. The validator
MUST reject `total >= 500_000_000`.

Checked-in profile counts:

| Profile | V | N | D | H | F | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| `smoke` | 256 | 2 | 64 | 4 | 192 | 123,200 |
| `arithmetic-30m` | 4,096 | 8 | 512 | 8 | 1,536 | 29,368,832 |
| `arithmetic-curriculum-30m` | 1,024 | 8 | 512 | 8 | 1,536 | 27,795,968 |
| `focused-125m` | 8,192 | 16 | 768 | 12 | 2,304 | 129,000,192 |

## Configuration invariants

- The trained tokenizer artifact size MUST be less than or equal to model
  `vocab_size`, and every emitted token ID MUST be inside the model vocabulary.
  A restricted BPE trainer MAY stop below its configured cap when no eligible
  merge remains; unused model output IDs are reported capacity, not text tokens.
- `max_seq_len` MUST equal the packed dataset block size and serving prompt cap.
- `pad_token_id` MUST be excluded from loss.
- `tie_embeddings` MUST be true for spec version 1.
- `norm` MUST be `rmsnorm` and `mlp` MUST be `swiglu` for spec version 1.
- Any change that affects parameter count, tensor names, checkpoint layout, or
  logits MUST change `model.format_version`.
