# Retail Bank Servicing Agent

This repository contains one canonical synthetic retail-bank conversational
pipeline:

- IBM Granite 4.1 8B base-tool SFT from
  `ibm-granite/granite-4.1-8b` revision
  `1504002f650e656a0a3789d99574df12e3e94ed0`;
- a second Granite servicing-remediation SFT stage added after observed
  multi-turn conversation and tool-use failures in the POC;
- a history-aware DistilBERT cross-encoder router with domain, servicing
  capability, and conversation-relation heads;
- exact frozen tool/final-response evaluation;
- an authenticated Gradio/ZeroGPU POC backed by a synthetic SQLite bank.

There are no real customers or banking connections. Every account, card,
transaction, transfer, and service case is fictional.

## Released Artifacts

| Component | Public artifact | Immutable revision |
|---|---|---|
| Granite servicing agent | [spkc83/retail-bank-servicing-agent-9b](https://huggingface.co/spkc83/retail-bank-servicing-agent-9b) | `1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| Stage-1 Granite tool-use checkpoint | [spkc83/retail-bank-agent-9b](https://huggingface.co/spkc83/retail-bank-agent-9b) | `085df3d089cfadd77424b548542da0390a54a23e` |
| Initial tool-use SFT data | [spkc83/retail-bank-agent-sft](https://huggingface.co/datasets/spkc83/retail-bank-agent-sft) | `183e7e1ed1aba9c3d7155e7b83b64dc854935055` |
| Servicing-remediation SFT data | [spkc83/retail-bank-servicing-alignment-sft](https://huggingface.co/datasets/spkc83/retail-bank-servicing-alignment-sft) | `0ce32f9c7a3edff227005e5b89b089947b87625a` |
| Prompt-identical training data revision | [spkc83/retail-bank-servicing-alignment-sft](https://huggingface.co/datasets/spkc83/retail-bank-servicing-alignment-sft) | `fea8aa1cda716954eb7322325e2be25c9f570ea3` |
| History-aware router | [spkc83/retail-bank-conversation-router](https://huggingface.co/spkc83/retail-bank-conversation-router) | `9e090c0fa21cebbaa03a431a7ce61e656c0739fe` |
| Router data | [spkc83/retail-bank-conversation-router-data](https://huggingface.co/datasets/spkc83/retail-bank-conversation-router-data) | `e9a64a2e7f2b622d5412c15eac4618ceca2150da` |
| ZeroGPU application | [retail-bank-servicing-poc](https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc) | See the Space diagnostics panel |

The released training and evaluation jobs used source revision
`475dc2b563ef87fa0c9aa597b0b0465d56d2ee0f`; the canonical pipeline pins its
current executable source in `configs/retail-bank-release.toml`. The standalone
application source is published at
[spkc83/retail-bank-servicing-poc](https://github.com/spkc83/retail-bank-servicing-poc).

## Request Flow

```text
authenticated synthetic customer
  -> CPU history-aware router
     -> high-confidence OOD: fixed scope response
     -> in-domain or uncertain: Granite 8.79B on ZeroGPU
        -> direct conversational response, or
        -> Granite tagged-JSON tool call
           -> synthetic SQLite tool execution
           -> correlated result returned to Granite
           -> Granite-authored grounded response
```

The router's domain decision and conversation-relation probabilities control
whether a turn reaches Granite; capability predictions are diagnostics only.
None of the classifier outputs enter the Granite prompt, select tools,
authorize actions, or provide arguments. The model receives token-budgeted
interaction groups and owns conversation, clarification, tool selection,
public arguments, and final wording.

## Start Here

The documentation is ordered so a junior developer can reproduce the system
without reading the implementation first:

1. [System overview](docs/01-system-overview.md)
2. [Data generation](docs/02-data-generation.md)
3. [Granite architecture and PEFT](docs/03-model-and-peft.md)
4. [Training, continuation, and recovery](docs/04-training-and-recovery.md)
5. [Conversation router](docs/05-dual-head-router.md)
6. [Frozen evaluation](docs/06-evaluation.md)
7. [Inference and ZeroGPU POC](docs/07-inference-and-poc.md)
8. [End-to-end runbook](docs/08-end-to-end-runbook.md)
9. [Conversation Router v4 release](docs/09-conversation-router-v4.md)
10. [Granite Servicing Alignment v4](docs/10-servicing-alignment-v4.md)
11. [Code/file map](docs/reference/file-map.md) and
   [artifact ledger](docs/reference/artifacts.md)

## Local Quick Start

Install the root development and training dependencies:

```bash
uv sync --extra dev --extra scale
```

Generate a small, fully validated synthetic tool-use corpus:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/prepare_tool_sft_data.py \
  --output-dir /tmp/retail-bank-tool-sft-smoke \
  --pilot-count 120
```

Regenerate the released history-aware router splits:

```bash
PYTHONPATH=src uv run python \
  scripts/retail_bank/prepare_conversation_router_data.py
```

Regenerate the composite Granite servicing-remediation corpus:

```bash
PYTHONPATH=src uv run python \
  scripts/retail_bank/prepare_servicing_alignment_data.py
```

Train the router locally without publishing:

```bash
PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py
```

Inspect the complete release plan without allocating a GPU, publishing, or
submitting a job:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/run_release_pipeline.py \
  --stage all
```

The canonical entry point contains data preparation, the two sequential Granite
SFT stages, router training, frozen evaluation, and deployment. Execution is
deliberately one stage at a time so each newly published immutable revision can
be captured before it is passed downstream.

Run the local quality gates:

```bash
PYTHONPATH=src uv run pytest -q tests
uv run ruff check .
MYPYPATH=src uv run mypy src scripts tests
uv lock --check
```

The paid Hugging Face Jobs commands, checkpoint retention policy, and recovery
paths are documented in
[training and recovery](docs/04-training-and-recovery.md). They are
intentionally separate from safe local preparation.

## Release Facts

The final servicing-remediation training job was
`spkc83/6a6ca6276b79c09949c1d6cb`. It ran for about 18 minutes 59 seconds on the
authorized paid GPU path at an estimated cost of about `$0.87`. The reported
training loss was `0.0069123295`, evaluation loss was `0.0002181597`, and token
accuracy was `0.999976121`.

The exact frozen evaluation job was `spkc83/6a6caac1a00abefd4b289b14`. It
evaluated 1,374 records and passed:

- tool names and arguments: `796/796`;
- executable tool trajectories: `700/700`;
- dependent multi-tool sequences: `96/96`;
- clarifications: `63/63`;
- banking FAQ answers: `258/258`;
- OOD paths: `35/35`;
- grounded factual responses: `1,141/1,141`;
- malformed calls, unsupported/private arguments, credential requests,
  in-domain false refusals, and OOD false accepts: `0`.

The corrected dataset revision `0ce32f9c7a3edff227005e5b89b089947b87625a` is the
published data identity. The training run used
`fea8aa1cda716954eb7322325e2be25c9f570ea3`; the rescore is valid because the
corrected rows are prompt-identical for generation and scoring. It is a rescore
of equivalent prompts, not a second generation run.

## Repository Map

```text
configs/        Granite PEFT configuration
data/sources/   pinned source and release-digest locks
data_cards/     dataset-card sources
docs/           canonical implementation and reproduction guide
model_cards/    model-card sources
poc/            standalone authenticated Gradio/ZeroGPU application
scripts/        data, training, recovery, evaluation, and Hub job entry points
src/hello_slm/  reusable corpus, tool-wire, evaluator, and router modules
tests/          root regression and documentation-contract tests
```

Repository code and the synthetic generative corpus are MIT licensed. Upstream
models and classifier datasets retain their own licenses.
