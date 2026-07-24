# Training specification

This document is normative for `hello-slm train`.

## Objective

Training optimizes next-token cross entropy over assistant response tokens from
the restricted chat corpus. The implementation MUST initialize the model from
random weights and MUST NOT download or load pretrained artifacts.

## Effective batch

The effective batch is:

```text
tokens_per_optimizer_step =
  micro_batch_size * max_seq_len * gradient_accumulation_steps * world_size
```

`world_size = 1` for this repository. Distributed training is out of scope. If
`drop_last = true`, incomplete final microbatches are skipped. Otherwise they
contribute their actual non-pad, loss-bearing token count to metrics.

## Optimizer

Use AdamW with decoupled weight decay.

- `betas = [0.9, 0.95]`
- `eps = 1e-8`
- `weight_decay = 0.1` for target runs, lower in smoke for stability
- Apply weight decay to linear and embedding weights.
- Do not apply weight decay to RMSNorm scale parameters.

Gradients are accumulated in FP32. Clip global gradient norm after accumulation
and before the optimizer step. Non-finite loss or gradients MUST abort training
without publishing a checkpoint as latest.

## Learning-rate schedule

The scheduler is warmup plus cosine decay by optimizer step:

```text
if step < warmup_steps:
  lr = max_lr * (step + 1) / warmup_steps
else:
  progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
  lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
```

`total_steps` is explicit in config. Epoch count is derived from
`total_steps * tokens_per_optimizer_step / train_loss_tokens` and recorded for
reporting; it does not silently change `total_steps`.

## Precision and devices

Smoke training MUST run on CPU with `float32`. Target training SHOULD use CUDA
with `bfloat16` autocast when hardware supports it, keeping master weights,
optimizer state, gradient accumulation, loss scaling checks, and checkpoint
serialization in FP32-compatible tensors. If requested precision is unavailable,
validation MUST fail unless `allow_precision_fallback = true`.

## Seeding and determinism

The config contains separate integer seeds for corpus split, tokenizer training,
dataset packing, model initialization, dataloader shuffling, and generation.
Training MUST set Python, NumPy if installed, and PyTorch RNGs. CUDA runs SHOULD
request deterministic algorithms where practical and MUST record any
non-deterministic backend flags in the train report.

## Validation and checkpoint cadence

Training MUST evaluate the validation split every `eval_interval_steps`, before
the first optimizer step when `eval_at_step_zero = true`, and at final step.
Checkpoints are written every `checkpoint_interval_steps` and at final step.
`latest.pt` is updated only after the step checkpoint is fully written and its
hash is recorded.

Each validation report includes:

- global optimizer step and consumed loss-bearing train tokens;
- train loss over the most recent logging window;
- validation loss and perplexity over all validation loss-bearing tokens;
- current learning rate and gradient norm;
- tokens per second and elapsed seconds;
- config, corpus, tokenizer, dataset, source, and environment fingerprints.

## Atomic checkpoint contents

A checkpoint is a single PyTorch file written to a temporary path, fsynced when
supported, and atomically renamed. A JSON integrity sidecar containing its byte
count and SHA-256 MUST be written before the checkpoint becomes loadable. It MUST
contain:

- `format_version`;
- model config snapshot and exact parameter count;
- `model_state_dict`;
- `optimizer_state_dict`;
- `scheduler_state_dict`;
- AMP scaler state when enabled;
- global optimizer step, microstep within accumulation, epoch estimate, consumed
  examples, consumed total tokens, and consumed loss-bearing tokens;
- Python, NumPy if installed, PyTorch CPU, and PyTorch CUDA RNG states;
- dataloader sampler state or deterministic cursor sufficient to resume without
  reshuffling;
- effective config canonical JSON and SHA-256;
- corpus manifest hash, normalized corpus hash, tokenizer hash, dataset shard
  hashes, and chat template hash;
- source commit if available, dependency inventory, platform, device, dtype, and
  backend determinism flags.

Checkpoint and dataset readers MUST verify the sidecar/manifest digest before
deserialization and MUST use PyTorch's restricted `weights_only` loader. Payloads
therefore contain tensors and primitive containers only.

## Strict resume

`hello-slm train --resume PATH` MUST load all checkpoint state before taking an
optimizer step. Resume MUST reject:

- changed effective config hash;
- changed corpus, tokenizer, dataset, or chat template hash;
- parameter-count mismatch;
- missing optimizer, scheduler, scaler, RNG, or sampler state;
- checkpoint format version unknown to the implementation;
- requested `max_seq_len`, tokenizer size, or special-token IDs that differ from
  the checkpoint.

The only allowed resume mutation is a smaller command-line `--max-steps` used by
tests to stop earlier than the config's `total_steps`; it MUST NOT change the
stored effective config hash.

## Failure handling

Training fails with exit code `2` for invalid config, invalid data references,
parameter cap violation, incompatible resume, or unavailable required precision.
It fails with exit code `1` for unexpected runtime errors. Partial temporary
outputs remain diagnostic only and MUST NOT be referenced by `latest.pt` or a
success manifest.

## Compute estimates

A rough dense-transformer training estimate is:

```text
training_flops ~= 6 * parameter_count * trained_tokens
activation_memory ~= micro_batch_size * max_seq_len * D * N * bytes_per_activation * safety_factor
```

The smoke profile trains only enough to prove plumbing. The focused profile is
credible as a small restricted-domain experiment, not as a general assistant:

| Profile | Params | Context | Effective tokens/step | Steps | Approx train tokens | Approx FLOPs |
|---|---:|---:|---:|---:|---:|---:|
| `smoke` | 123,200 | 128 | 512 | 8 | 4,096 | 3.0e9 |
| `arithmetic-30m` | 29,368,832 | 640 | 40,960 | 2,000 | 81.9M | 1.44e16 |
| `arithmetic-curriculum-30m` | 27,795,968 | 128 | 16,384 | 2,000 | 32.8M | 5.47e15 |
| `focused-125m` | 129,000,192 | 512 | 262,144 | 20,000 | 5.24e9 | 4.06e18 |

The target estimate is a planning number. A release claim requires actual train
logs, validation metrics, restore drill evidence, and the evaluation gates.
