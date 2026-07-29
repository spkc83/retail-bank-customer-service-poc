# Retail Bank Agent

This repository contains one end-to-end experiment:

- a PEFT-finetuned IBM Granite conversational agent with 8,791,592,960
  parameters;
- a DistilBERT dual-head classifier for banking-domain/OOD detection and
  Banking77 intent diagnostics;
- governed data generation, training, continuation, recovery, merge, and
  frozen-evaluation code;
- a public Gradio application that runs the Granite model on Hugging Face
  ZeroGPU and executes model-generated calls against a synthetic bank.

There are no real customers or banking connections. Every account, card,
transaction, transfer, and service case is fictional.

## Released system

| Component | Public artifact | Immutable revision |
|---|---|---|
| Granite agent | [spkc83/retail-bank-agent-9b](https://huggingface.co/spkc83/retail-bank-agent-9b) | `085df3d089cfadd77424b548542da0390a54a23e` |
| Tool-use SFT data | [spkc83/retail-bank-agent-sft](https://huggingface.co/datasets/spkc83/retail-bank-agent-sft) | `183e7e1ed1aba9c3d7155e7b83b64dc854935055` |
| Dual-head classifier | [spkc83/retail-bank-domain-intent-router](https://huggingface.co/spkc83/retail-bank-domain-intent-router) | `136ee159d19cda7f585dd122907bbeb1ef4ec4db` |
| Classifier data | [spkc83/retail-bank-router-training-data](https://huggingface.co/datasets/spkc83/retail-bank-router-training-data) | `54ff186a03501d76dc643dbed3d82729267ce811` |
| ZeroGPU application | [retail-bank-servicing-poc](https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc) | See the Space diagnostics panel |

The standalone application source is also published at
[spkc83/retail-bank-servicing-poc](https://github.com/spkc83/retail-bank-servicing-poc).

## Request flow

```text
authenticated synthetic customer
  -> CPU dual-head classifier
     -> high-confidence OOD: fixed scope response
     -> in-domain or uncertain: Granite 8.79B on ZeroGPU
        -> direct conversational response, or
        -> Granite tagged-JSON tool call
           -> synthetic SQLite tool execution
           -> correlated result returned to Granite
           -> Granite-authored grounded response
```

The intent head is diagnostic. Its predictions do not enter the generation
prompt, select tools, or provide arguments. The model receives complete,
token-budgeted interaction groups and owns conversation, clarification, tool
selection, public arguments, and final wording.

## Start here

The documentation is ordered so a junior developer can reproduce the system
without reading the implementation first:

1. [System overview](docs/01-system-overview.md)
2. [Data generation](docs/02-data-generation.md)
3. [Granite architecture and PEFT](docs/03-model-and-peft.md)
4. [Training, continuation, and recovery](docs/04-training-and-recovery.md)
5. [Dual-head router](docs/05-dual-head-router.md)
6. [Frozen evaluation](docs/06-evaluation.md)
7. [Inference and ZeroGPU POC](docs/07-inference-and-poc.md)
8. [End-to-end runbook](docs/08-end-to-end-runbook.md)
9. [Code/file map](docs/reference/file-map.md) and
   [artifact ledger](docs/reference/artifacts.md)

## Local quick start

Install the root development and training dependencies:

```bash
uv sync --extra dev --extra scale
```

Generate a small, fully validated synthetic tool-use corpus:

```bash
PYTHONPATH=src uv run python scripts/banking_v2/prepare_tool_sft_data.py \
  --output-dir /tmp/retail-bank-tool-sft-smoke \
  --pilot-count 120
```

Regenerate the released dual-head classifier splits from checksum-pinned
Banking77 and CLINC sources. The command fails if any released split digest
changes:

```bash
PYTHONPATH=src uv run python scripts/banking_v2/prepare_dual_head_router_data.py
```

Inspect the Granite training plan without allocating a GPU or submitting a
job:

```bash
PYTHONPATH=src uv run python scripts/banking_v2/cloud_train_tool_sft.py \
  --manifest data/banking-v3-tool-sft/manifest.json
```

Run the local quality gates:

```bash
PYTHONPATH=src uv run pytest -q tests
uv run ruff check .
MYPYPATH=src uv run mypy src scripts tests
uv lock --check
```

The paid Hugging Face Jobs commands, checkpoint persistence requirements, and
recovery paths are documented in
[training and recovery](docs/04-training-and-recovery.md). They are
intentionally separate from the safe local quick start.

## Release facts

The generative corpus contains 9,000 self-authored synthetic conversations:
6,304 train, 1,349 validation, and 1,347 frozen test. It covers all nine public
mock-bank tools, success and error results, clarification, FAQ, OOD,
hard-negative private-field requests, multi-turn context, and dependent
multi-tool sequences.

The released model is a merged FP16 adaptation of
`ibm-granite/granite-4.1-8b` at base revision
`1504002f650e656a0a3789d99574df12e3e94ed0`. Training uses BF16 LoRA with rank
32, alpha 64, dropout 0.05, a 2,048-token maximum sequence, and attention plus
MLP projection targets. The retained adapter is published separately under the
model repository's `adapter/` directory.

The frozen 1,347-record evaluation passed 774/774 tool names and arguments,
678/678 executable trajectories, 96/96 dependent multi-tool sequences, 63/63
clarifications, 258/258 FAQs, 30/30 OOD paths, and 1,119/1,119 grounded facts.

## Repository map

```text
configs/        Granite PEFT configuration
data/sources/   pinned classifier source and release-digest lock
data_cards/     published dataset-card sources
docs/           canonical implementation and reproduction guide
model_cards/    published model-card sources
poc/            standalone authenticated Gradio/ZeroGPU application
scripts/        data, training, recovery, evaluation, and Hub job entry points
src/hello_slm/  reusable corpus, tool-wire, evaluator, and router modules
tests/          root regression and documentation-contract tests
```

Repository code and the synthetic generative corpus are MIT licensed. Upstream
models and classifier datasets retain their own licenses.
