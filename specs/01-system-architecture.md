# System architecture

This document is normative. `MUST`, `MUST NOT`, `SHOULD`, and `MAY` have their
usual requirements-language meanings.

## Pipeline

```text
declared JSONL corpus
        │
        ▼
validate + normalize + conversation split
        │
        ├──────────────► corpus report / fingerprints
        ▼
train restricted tokenizer
        │
        ├──────────────► tokenizer.json / vocabulary report
        ▼
render chat template + causal token packing
        │
        ├──────────────► train/validation/test token shards
        ▼
initialize decoder-only Transformer from random weights
        │
        ▼
train ──► atomic checkpoints ──► resume
        │
        ▼
evaluate quality + safety + memorization
        │
        ▼
release bundle ──► bounded local chat CLI
```

## Source of truth and precedence

Normative behavior is defined in this order:

1. JSON schemas define serialized input shape.
2. TOML configuration defines an experiment's effective values.
3. This specification set defines invariants and semantics.
4. The reference implementation realizes those contracts.

If code and a specification disagree, validation MUST fail until one is
deliberately versioned. Silent fallback is forbidden.

## Stage contract

Every stage MUST:

- accept only paths and settings from the effective configuration;
- validate all upstream artifact format versions and SHA-256 fingerprints;
- write to a temporary path and atomically rename complete artifacts;
- emit a machine-readable manifest containing inputs, outputs, counts, hashes,
  seed, implementation version, start/end time, and status;
- be deterministic for a fixed corpus, configuration, dependency set, and device
  class, subject to documented PyTorch limitations;
- fail non-zero without marking partial output complete.

## Artifact layout

```text
artifacts/<run-id>/
  effective-config.json
  run-manifest.json
  reports/{corpus,tokenizer,train,eval}.json
  tokenizer/tokenizer.json
  dataset/{train,validation,test}.pt
  checkpoints/step-<N>.pt
  release/{config.json,model.pt,tokenizer.json,model-card.md}
```

Generated artifacts are intentionally ignored by Git. Human-authored configs,
schemas, specs, tests, and the synthetic example corpus are version controlled.

## CLI contract

The package exposes `hello-slm` with these subcommands:

| Command | Required result |
|---|---|
| `validate --config PATH` | Validate config, schemas, corpus declaration, and exact model parameter limit without allocating the target model. |
| `build-tokenizer --config PATH` | Validate/normalize the corpus and train only from configured training conversations. |
| `build-dataset --config PATH` | Render chats, encode, split, and pack fixed-length causal examples. |
| `train --config PATH [--resume PATH]` | Initialize from scratch or resume exactly, optimize, and checkpoint. |
| `eval --config PATH --checkpoint PATH` | Compute configured held-out and safety metrics without mutating weights. |
| `chat --config PATH --checkpoint PATH` | Run bounded local generation with the normative chat template. |

Exit `0` means success, `2` means invalid user/config/data input, and `1` means
an unexpected runtime failure. Commands MUST NOT download data or models.

## Configuration profiles

- `configs/smoke.toml` is the executable acceptance fixture. It prioritizes speed
  and structural verification, not model quality.
- `configs/focused-125m.toml` is the build specification for a useful-scale
  experiment. It uses the synthetic manifest only so clean-checkout structural
  validation is executable. Full corpus validation intentionally fails until a
  real run replaces that manifest with an approved target corpus whose splits
  match `seeds.corpus_split` and passes every data/evaluation gate. It MUST
  remain below the hard parameter cap but is not trained in CI.

## Versioning

All serialized artifacts carry `format_version = 1`. Breaking changes increment
the version. Readers MUST reject unknown major versions. Configuration and corpus
fingerprints use canonical JSON encoding followed by SHA-256.
