# hello-SLM

An executable hello-world blueprint for training a focused conversational small
language model from random initialization on a closed corpus and restricted
vocabulary. It contains the full specification set plus a deliberately small
PyTorch reference pipeline.

Two profiles share the same implementation:

- `smoke`: 123,200 parameters, tiny synthetic MIT-licensed conversations, CPU
  execution, and structural acceptance only;
- `focused-125m`: 129,000,192 parameters, intended as the starting configuration
  for an approved domain corpus and a real GPU run.

The arithmetic work adds two profiles:

- `arithmetic-30m`: 29,368,832 parameters, a 4,096-token restricted vocabulary,
  and a reproducibly prepared 50,000-example Orca Math baseline corpus;
- `arithmetic-curriculum-30m`: 27,795,968 parameters, a digit-atomic restricted
  vocabulary, and 50,000 deterministic, answer-verified conversations.

Those original hello-world profiles are below the hard limit of 500,000,000
parameters. Banking-v2 is a separate, larger domain-adaptation experiment and
does not satisfy that original size/from-scratch contract.

## What is included

- normative data governance, tokenizer, model, training, evaluation, safety,
  serving, reproducibility, operations, and verification specs;
- versioned TOML configurations and JSON schemas;
- a synthetic conversation corpus with exact hashes and training consent;
- from-scratch restricted BPE, packed causal chat datasets, a decoder-only
  RoPE/RMSNorm/SwiGLU Transformer, training/checkpoint/resume, evaluation, and
  bounded generation;
- unit and end-to-end smoke tests.

Start with [the specification index](specs/README.md) and the
[requirements ledger](specs/00-requirements.md).

## Quick start

Requires Python 3.11+ and PyTorch. Training stages never download a model,
tokenizer, or corpus; source acquisition is an explicit preparation step.

```bash
python -m pip install -e '.[dev]'
hello-slm validate --config configs/smoke.toml
hello-slm smoke --config configs/smoke.toml
hello-slm chat --config configs/smoke.toml \
  --checkpoint artifacts/smoke/checkpoints/latest.pt \
  --prompt "Hello"
```

## Prepare the arithmetic corpus

Install the data-preparation extra, then acquire and transform the immutable
Orca Math source. Raw and generated files are ignored by Git; the source lock
and preparation code are versioned.

```bash
python -m pip install -e '.[data,dev]'
python -m hello_slm.prepare_math
hello-slm validate --config configs/arithmetic-30m.toml
```

The default preparation scans all 200,035 source rows and deterministically
selects 50,000 normalized, exact-deduplicated conversations (48,992 train, 503
validation, and 505 test with seed 2101). See
`data/arithmetic/preparation-report.json` for accepted/rejected counts and
source fingerprints. Tokenizer construction and model training are subsequent
explicit stages.

## Train the working arithmetic curriculum

The controlled curriculum is the primary hello-world tutor. It supports bounded
integer addition, subtraction, and exact division. Multiplication examples are
included as exploratory data, but multiplication is not a supported quality
claim for this checkpoint.

```bash
python -m hello_slm.prepare_arithmetic_curriculum
hello-slm validate --config configs/arithmetic-curriculum-30m.toml
hello-slm build-tokenizer --config configs/arithmetic-curriculum-30m.toml
hello-slm build-dataset --config configs/arithmetic-curriculum-30m.toml
hello-slm train --config configs/arithmetic-curriculum-30m.toml
hello-slm eval --config configs/arithmetic-curriculum-30m.toml
hello-slm eval-arithmetic --config configs/arithmetic-curriculum-30m.toml
hello-slm chat --config configs/arithmetic-curriculum-30m.toml \
  --prompt "What is 2 + 3?" --max-new-tokens 64
```

The completed 2,000-step run reached test-split perplexity `1.0115` and token
accuracy `99.54%`. Deterministic generated-answer evaluation scored `95.10%`
over supported operations: addition `100%`, subtraction `100%`, and exact
division `83.72%`. This is a closed-corpus paraphrase/fact-recall check: paired
phrasings of a fact may cross splits, so it is not evidence of arithmetic
reasoning on unseen facts. Multiplication scored `30%` and remains explicitly
unsupported.

## Shiny arithmetic chat lab

The local Shiny for Python app loads the completed arithmetic checkpoint lazily,
keeps it in memory between prompts, and includes preset addition, subtraction,
exact-division, and limitation-check cases. Presets and typed prompts use the
same deterministic inference path.

```bash
python -m pip install -e '.[app]'
shiny run --reload src/hello_slm/shiny_app.py
```

Open the local address printed by Shiny (normally `http://127.0.0.1:8000`). The
app requires the generated tokenizer and trained checkpoint under
`artifacts/arithmetic-curriculum-30m/`; those large local artifacts remain
ignored by Git. Each displayed turn is evaluated independently rather than as
multi-turn context.

## Banking-v2 adaptation track

The arithmetic checkpoint produces gibberish outside its presets because it is
a 27.8M-parameter model trained on a small, highly regular arithmetic corpus
with a 156-token realized vocabulary. Its strong closed-corpus score measures
paraphrase/fact recall, not broad language understanding.

Banking-v2 addresses that limitation as a separate experiment:

- 24,042 prepared generative conversations: 24,006 Bitext banking QA records
  plus 36 self-authored OOD and multi-turn records;
- 22,033 train, 1,008 validation, and 1,001 test conversations, split by whole
  near-duplicate groups;
- 13,083 Banking77 rows reserved for router evaluation and never used as
  generative SFT;
- zero unresolved placeholders, detected PII-like strings, or normalized
  cross-split user duplicates in the generated report;
- corpus fingerprint
  `6d26ef95cdfcc16bdb1056844062f6b93618b5092626058610ccf0502352727f`;
- exact out-of-domain response enforced by the serving router before
  generation;
- session-isolated, bounded multi-turn history and an evaluation-gated
  one-versus-four candidate test-time-scaling policy.

Prepare and validate the local dataset:

```bash
python -m pip install -e '.[data,scale,dev]'
python -m hello_slm.banking_data audit-sources
python -m hello_slm.banking_data prepare
python -c "from pathlib import Path; from hello_slm.banking_data import validate_banking_v2_manifest; validate_banking_v2_manifest(Path('data/banking-v2/manifest.json'))"
```

The proposed model upcycles the pinned Qwen2.5 1.5B Instruct checkpoint into a
Qwen2-MoE model with 8,943,713,792 total parameters and approximately
2,073,443,840 active parameters per token. This is supervised domain adaptation,
not from-scratch 9B pretraining: the available corpus is far too small to
pretrain a capable 9B general language model from random initialization.

The local code/specification and tiny MoE backward smoke tests do not mean the
9B checkpoint has been trained. The authorized paid run uses one RTX PRO 6000,
has a 5-hour/USD 13.75 cap, and targets the private Hub repository
`spkc83/retail-bank-servicing-moe-9b`. Paid compute and external repository creation are
approval-gated.

Inspect the guarded job plan and run the offline tiny training loop:

```bash
python scripts/banking_v2/cloud_train_banking_moe.py
python scripts/banking_v2/cloud_train_banking_moe.py \
  --run-tiny-smoke --max-steps 1 --output-dir artifacts/banking-v2-tiny-smoke
```

The full worker requires all three remote guards: `--execute-remote`,
`--allow-remote-execution`, and
`HELLO_SLM_ALLOW_REMOTE_TRAINING=banking-v2`. It saves the inference artifact at
`artifacts/banking-v2-moe-9b/final/` and may upload it only when
`--push-to-hub` is explicitly supplied.

The separate banking Shiny app keeps bounded conversation history per session,
shows router confidence and candidate count, and includes banking, follow-up,
and OOD presets:

```bash
shiny run --reload src/hello_slm/banking_shiny_app.py
```

OOD routing works without a checkpoint. In-domain inference fails explicitly
until the trained model exists at `artifacts/banking-v2-moe-9b/final/`; set
`HELLO_SLM_BANKING_MODEL` to use another local Transformers checkpoint.

See [the banking-v2 specification](specs/banking-v2/README.md) for data,
conversion, expert-health, evaluation, and serving gates.

## Model-driven retail-bank POC

The standalone
[`retail-bank-customer-service-poc`](poc/retail-bank-customer-service-poc)
turns the banking artifacts into an authenticated, end-to-end demonstration:

- two static demo logins mapped to separate fictional customers;
- a CPU dual-head OOD and Banking77 intent gate;
- native Qwen tool calls generated by the 9B checkpoint on ZeroGPU;
- server-side tool validation and explicit authorization for write actions;
- per-browser-session in-memory SQLite cloned from immutable synthetic data;
- model-authored final responses grounded in the executed tool result.

The standalone source is published at
https://github.com/spkc83/retail-bank-customer-service-poc and the live
application is at
https://huggingface.co/spaces/spkc83/retail-bank-customer-service-poc.
Neither connects to a real bank or performs real transactions.

Individual stages are also available:

```bash
hello-slm build-tokenizer --config configs/smoke.toml
hello-slm build-dataset --config configs/smoke.toml
hello-slm train --config configs/smoke.toml --max-steps 2
hello-slm eval --config configs/smoke.toml \
  --checkpoint artifacts/smoke/checkpoints/latest.pt
```

## Important boundary

Passing the smoke flow proves that the pipeline is coherent; it does not produce
a useful, factual, private, or safe assistant. This repository does not include
production serving, moderation, distributed training, formal privacy, or broad
safety certification. A real release requires an approved corpus, target-scale
training evidence, the numeric evaluation gates, and completed data/model cards.

## Repository map

```text
configs/        experiment and evaluation profiles
data/sources/   immutable external and synthetic dataset locks
examples/       synthetic restricted corpus
schemas/        serialized contract schemas
specs/          normative build and acceptance specifications
templates/      required model/data card templates
src/hello_slm/  minimal reference implementation
tests/          unit and end-to-end verification
```

Licensed under MIT. The synthetic corpus is authored for this repository and is
covered by the corpus manifest's explicit training grant.
