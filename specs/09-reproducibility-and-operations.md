# Reproducibility and operations specification

This document defines run records, resource planning, packaging, and recovery
expectations for the hello-world and focused 125M profiles.

## Deterministic records

Every command MUST write a manifest with:

- command name and arguments;
- effective configuration SHA-256;
- source code version or working-tree marker;
- Python, PyTorch, operating system, and device metadata;
- random seeds for Python, NumPy if used, and PyTorch;
- corpus, tokenizer, dataset, checkpoint, and evaluation digests;
- start time, end time, duration, status, and error category.

For CUDA runs, manifests MUST include GPU name, driver/runtime versions, enabled
precision, deterministic algorithm settings, and a note that bitwise identical
results are not guaranteed across device classes.

## Configuration snapshots

Each run MUST serialize the fully resolved configuration as canonical JSON at
`artifacts/<run-id>/effective-config.json`. Resume MUST reject changed effective
configuration, tokenizer digest, dataset digest, model architecture digest, or
optimizer state unless an explicit future non-reproducible override is added.

## Hardware and compute planning

| Profile | Model size | Minimum hardware | Expected use |
|---|---:|---|---|
| `smoke` | under 1M parameters | CPU, 2 GB RAM | CI and local structural checks. |
| `arithmetic-30m` | 29.4M parameters | 1 GPU with 8-12 GB VRAM, or CPU for slow experiments | Focused grade-school arithmetic training. |
| `arithmetic-curriculum-30m` | 27.8M parameters | 1 GPU with 4 GB free VRAM | Bounded verified arithmetic curriculum training. |
| `focused_125m` | roughly 100-150M parameters, always `<500M` | 1 GPU with at least 24 GB VRAM, or slower CPU experimentation with reduced batch size | Real focused SLM experiment. |

The target profile SHOULD use mixed precision on compatible GPUs, gradient
accumulation, checkpointing at fixed step intervals, and validation at fixed
token intervals. Distributed training is out of scope for this repository.

## Resource limits

Commands MUST expose configuration for:

- maximum input files and bytes;
- maximum tokenizer vocabulary size;
- maximum sequence length;
- maximum train steps or tokens;
- checkpoint interval and retention count;
- evaluation generation limits;
- artifact directory size warning threshold.

Validation MUST reject a model with `>= 500,000,000` total parameters before
training allocation. The parameter report MUST include embedding, attention,
feed-forward, normalization, and output-head counts.

## Checkpoints and recovery

Checkpoints MUST contain:

- model state;
- optimizer state;
- scheduler state if configured;
- gradient scaler state if mixed precision is used;
- RNG state;
- completed step and token count;
- effective configuration digest;
- dataset cursor or sampler state;
- checkpoint file digest written after atomic completion.

Training resume MUST reproduce the next batch order for the same device class
and reject partial checkpoints. A failed write MUST leave the previous complete
checkpoint usable.

## Packaging

Release bundles MUST be written under:

```text
artifacts/<run-id>/release/
  config.json
  manifest.json
  model.pt
  tokenizer.json
  model-card.md
  data-card.md
  eval-report.json
  sample-transcript.jsonl
```

`config.json` MUST contain only serving-time settings and architecture metadata
needed for local loading. It MUST NOT contain private corpus paths, scan reports,
environment secrets, or trainer-only credentials.

## Observability

Training logs MUST include:

- step;
- tokens processed;
- train loss;
- validation loss and perplexity at validation intervals;
- learning rate;
- gradient norm when available;
- throughput in tokens per second;
- checkpoint path and digest when written.

Evaluation logs MUST include metric values, threshold values, pass/fail status,
and report path. Logs SHOULD be JSONL to support simple local inspection.

The project MUST NOT require a hosted telemetry service. Optional experiment
tracking MAY be added only when disabled by default and never sends corpus text,
secrets, or private scan snippets.

## Failure categories

Manifests MUST use one of these failure categories:

- `invalid_config`;
- `invalid_corpus`;
- `integrity_mismatch`;
- `resource_limit`;
- `runtime_error`;
- `threshold_failure`;
- `interrupted`.

Interrupted training SHOULD leave the latest complete checkpoint available for
resume. Evaluation threshold failures MUST still write a complete report with
`release_eligible = false`.

## Production limitations

This repository does not provide:

- distributed training orchestration;
- hosted inference;
- authentication, authorization, or rate limiting;
- abuse monitoring;
- human escalation workflows;
- formal privacy guarantees;
- broad safety certification.

Any production deployment MUST add those controls outside this hello-world
reference project and MUST re-run evaluation on the final deployment stack.
