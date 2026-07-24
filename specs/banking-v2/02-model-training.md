# Banking-v2 model conversion and training

## Scope

Banking-v2 is domain adaptation of the pinned
`Qwen/Qwen2.5-1.5B-Instruct` checkpoint at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. It is not from-scratch
pretraining and is not covered by the original `<500M` hello-world contract.
The prepared banking corpus is suitable for supervised domain adaptation, not
for training a general 9B model from random initialization.

## Dense baseline

Before accepting the MoE result, train and evaluate a LoRA-adapted dense 1.5B
baseline with `configs/banking-v2-dense-adapter.toml`. The MoE must meet every
absolute evaluation gate and must not regress the held-out composite score
relative to this baseline.

## MoE shape

The model uses the Transformers Qwen2-MoE implementation:

- vocabulary: 151,936
- hidden size: 1,536
- layers: 28
- query heads / key-value heads: 12 / 2
- shared expert intermediate size: 8,960
- routed experts per layer: 28
- routed expert intermediate size: 2,048
- experts selected per token: 2, with normalized top-k routing
- router auxiliary-loss coefficient: 0.01
- tied token embeddings and language-model head

The exact effective parameter count is 8,943,713,792, with an estimated
2,073,443,840 parameters active for one token.

## Dense-to-MoE initialization

Copy all compatible non-MLP tensors from the pinned dense checkpoint. Initialize
the shared expert gate/up projections from the dense MLP, initialize its down
projection at twice the dense down projection, and initialize its sigmoid gate
at zero. This preserves the dense MLP output because `sigmoid(0) = 0.5`.

Initialize each routed expert's gate/up projections from deterministic aligned
slices of the dense tensors plus seed-controlled small noise. Initialize routed
down projections to zero so the residual MoE branch initially contributes
nothing. Initialize router weights with seed-controlled, small, nonzero normal
values. A conversion smoke test must demonstrate dense-versus-converted logit
equivalence at absolute tolerance `1e-5` before any optimizer update.

## Optimization

Use BF16, activation checkpointing, and full-shard FSDP on four 80GB A100 GPUs.
Freeze copied attention, normalization, embedding, shared-expert, and routed
gate/up weights. Train router weights and routed residual down projections.
Checkpoint every 250 optimizer steps. A resume must verify the base revision,
dataset fingerprint, converted-state manifest hash, optimizer, scheduler, and
random-number-generator state.

After the 250-step routing warm-up, stop a run that violates any expert-health
gate:

- every expert assignment fraction is at least 0.5%
- no expert assignment fraction exceeds 20%
- normalized routing entropy is at least 0.75
- router auxiliary loss is finite
- routed down projections receive nonzero finite gradients

## Compute and authorization

The proposed Hugging Face Jobs flavor is `a100x4` (four A100 GPUs, 320GB total).
The operator cap is 10 hours and USD 100. The intended destination is the
private Hub repository `spkc83/hello-banking-moe-9b`. Creating that repository
or starting paid compute requires explicit operator approval; local dry runs
must never imply that training occurred.
