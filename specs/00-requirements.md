# Requirements and acceptance ledger

Status values are `specified`, `implemented`, and `verified`. A requirement is
complete only when its evidence exists and its verification command passes.

| ID | Requirement | Acceptance evidence | Status |
|---|---|---|---|
| R-001 | Train a language model from scratch. | Tokenizer and weights are created only from the configured local corpus; no pretrained artifact is accepted by any command. | verified |
| R-002 | Keep every model below 500,000,000 parameters. | `hello-slm validate` computes the implementation's exact parameter count and rejects `>= 500_000_000`. | verified |
| R-003 | Optimize for focused conversational/chat use. | Corpus uses ordered chat turns, a chat template, causal loss masking, conversation-level splits, and chat-specific evaluation. | verified |
| R-004 | Use only an explicitly restricted corpus. | Corpus manifest identifies every input, its license/provenance, hash, allowed use, and exclusion decision; undeclared files are rejected. | verified |
| R-005 | Use a restricted vocabulary. | The tokenizer has a configured fixed maximum size, immutable special tokens, an allowed-character policy, deterministic training, and explicit unknown handling. | verified |
| R-006 | Provide a hello-world path. | A tiny synthetic corpus and smoke config execute tokenize → pack → train → save → resume → evaluate → generate on CPU. | verified |
| R-007 | Also provide a credible focused-SLM target. | A separate target config describes a roughly 100–150M parameter decoder model and passes structural validation without allocating or training it; full data validation requires an approved target manifest. | verified |
| R-008 | Make the pipeline reproducible. | Config snapshot/hash, seeds, corpus/tokenizer hashes, environment metadata, optimizer/scheduler/RNG state, and atomic checkpoints are specified and persisted. | verified |
| R-009 | Define quality gates. | Data, tokenizer, training, offline chat, safety, contamination, and release thresholds have explicit pass/fail rules. | specified |
| R-010 | Define safety, privacy, and security controls. | PII/secret screening, prompt/content policy, poisoning controls, artifact integrity, and release limitations are specified. | specified |
| R-011 | Provide build-ready interfaces. | Versioned TOML configs, JSON schemas, artifact layouts, CLI contracts, exit codes, and stage dependencies are normative. | verified |
| R-012 | Start implementing after specifications exist. | Runnable `validate`, `build-tokenizer`, `build-dataset`, `train`, `eval`, and `chat` commands plus tests exist after the spec set. | verified |
| R-013 | Do not imply production readiness. | README and model card state the example's limits; production scaling and distributed training are explicitly out of scope. | verified |
| R-014 | Provide a reproducible arithmetic-tutor corpus. | The pinned Orca Math source is hash-verified, fully scanned, normalized, exact-deduplicated, deterministically sampled/split, and accepted by full corpus validation. | verified |
| R-015 | Provide a working bounded arithmetic curriculum. | A self-authored deterministic corpus, digit-atomic tokenizer, fresh 2,000-step run, and generated final-integer gate verify addition, subtraction, and exact division; unsupported multiplication remains reported. | verified |

## Design decision

Three repository shapes were considered:

1. **Documentation only.** Small and framework-neutral, but contracts can drift
   and the hello-world path cannot be proven.
2. **Production training platform.** Operationally complete, but inappropriate
   for a teaching repository and impossible to validate without substantial
   compute and infrastructure.
3. **Executable specification plus minimal trainer (selected).** Normative
   contracts are paired with a small PyTorch reference implementation. The
   smoke configuration is runnable locally; the target configuration exercises
   the same validation and code paths without claiming a completed target run.

## Hard boundaries

- All training text MUST be present in the declared corpus manifest.
- Tokenizer training and model initialization MUST start from random state.
- A model with `>= 500_000_000` trainable plus non-trainable parameters MUST be
  rejected before allocation.
- Corpus splits MUST occur at conversation level before token packing.
- Checkpoint resume MUST reject a changed effective configuration or data/tokenizer
  fingerprint unless an explicit non-reproducible override is added in a future
  spec revision.
- This repository MUST NOT claim safety, factuality, or production readiness
  based on the synthetic smoke corpus.
