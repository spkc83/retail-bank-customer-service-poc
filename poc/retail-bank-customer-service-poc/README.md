---
title: Retail Bank Customer Service POC
emoji: 🏦
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
python_version: 3.10
app_file: app.py
pinned: false
suggested_hardware: zero-a10g
models:
  - spkc83/retail-bank-agent-9b
  - spkc83/retail-bank-domain-intent-router
datasets:
  - spkc83/retail-bank-agent-sft
  - spkc83/retail-bank-router-training-data
short_description: Model-driven 8.79B synthetic retail-bank service agent.
---

# Retail Bank Customer Service POC

This authenticated POC tests whether a dual-head OOD/intent classifier plus a
tool-trained 8.79B Granite model can provide natural multi-turn customer
service and operate a synthetic retail-bank backend.

Everything is fictional. The application has no connection to a bank and
cannot access real accounts or perform real transactions.

## Live artifacts

- Application: https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc
- POC source: https://github.com/spkc83/retail-bank-servicing-poc
- Model-development source: https://github.com/spkc83/retail-bank-servicing
- Model: https://huggingface.co/spkc83/retail-bank-agent-9b
- Tool-use dataset:
  https://huggingface.co/datasets/spkc83/retail-bank-agent-sft
- Dual-head classifier:
  https://huggingface.co/spkc83/retail-bank-domain-intent-router

## Runtime

```text
Authenticated synthetic customer and session transcript
  → one managed ZeroGPU chat event
  → CPU dual-head classifier
  → high-confidence OOD: governed scope response
  → in-domain or uncertain: 8.79B model generation
  → direct answer, clarification, or tagged-JSON tool calls
  → generated calls execute against session-isolated synthetic SQLite
  → results return to the same model for the final response
```

The intent head's top predictions are diagnostic metadata. They do not enter
the generation prompt, select a tool, or supply arguments. The 8.79B model owns greetings, conversation,
clarification, tool choice, public arguments, and final wording. The runtime
only budgets context, parses and validates the tagged-JSON wire format, invokes
the named mock function, and records diagnostics.

The live generation prompt and iterative model → tool → model protocol match
the SFT corpus and frozen evaluator. A first-pass answer without a tool call is
returned directly; the runtime does not add an untrained reflection or repair
pass.

## Conversation context

The application stores complete valid interaction groups: user messages,
assistant tool calls, correlated tool results, and final model responses.

Each inference uses an 8,192-token input budget and reserves 512 tokens for
generation. The current interaction and system instructions are retained
first; newest complete prior interaction groups are then added while they fit.
A tool chain is never split across the context boundary.

## Synthetic tools

- list accounts, cards, transactions, transfers, and service cases;
- freeze or replace a card;
- dispute a transaction by merchant description;
- cancel a pending transfer by recipient.

Calls execute in generated order. Schema or backend errors return to the model
as tool results so it can explain the outcome conversationally.

## Proving model inference

The diagnostics panel exposes:

- exact model repository and immutable revision;
- runtime and CUDA device;
- response path and model-call count;
- raw `base`, `grounded_final`, and iterative tool-follow-up outputs;
- generated tool names and public arguments;
- correlated tool results;
- prompt and output SHA-256 values for every model pass.

A successful live turn is counted as 8.79B inference only when diagnostics show
`spkc83/retail-bank-agent-9b` at revision
`b47e2028c8cf573eb50ef7fe1c48d67e2a08e865` and a CUDA device. Preset prompts
are evaluation cases, not hard-coded routes.

The pinned checkpoint passed a frozen 750-record test split with 100% tool-name
and argument accuracy, 100% executable-tool success, 39/39 exact dependent
multi-tool sequences, 48/48 appropriate clarifications, 100% grounded
factuality, and zero malformed calls, private arguments, or credential
requests.

If ZeroGPU allocation or generation fails, the UI reports model
unavailability. It does not substitute a Python-generated banking answer.

## Authentication

The demo usernames are `alex.demo` and `maya.demo`. Passwords come from the
Space's write-only `DEMO_AUTH_JSON` secret and are displayed on the login page
for public testing. Authentication only selects isolated synthetic records.

## Local verification

```bash
python -m pip install -r requirements.txt
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-chars"}'
export POC_SKIP_MODEL_LOAD=1
export POC_SKIP_ROUTER_LOAD=1
pytest -q
ruff check .
```

Release verification additionally requires live ZeroGPU read, write,
multi-tool, clarification, FAQ, OOD, and multi-turn cases with the exact model
revision visible in diagnostics.
