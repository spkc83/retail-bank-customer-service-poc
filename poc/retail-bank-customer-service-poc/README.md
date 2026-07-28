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
  - spkc83/retail-bank-servicing-moe-9b
  - spkc83/retail-bank-domain-intent-router
datasets:
  - spkc83/retail-bank-router-training-data
short_description: Authenticated 9B retail-banking demo with synthetic data.
---

# Retail Bank Customer Service POC

An authenticated, model-driven customer-service demonstration for a fictional
retail bank. The 9B banking model runs on Hugging Face ZeroGPU, proposes native
Qwen tool calls, receives validated results from a session-isolated SQLite
backend, and writes the final customer response.

Everything in the application is synthetic. It has no connection to a real
bank, cannot access real accounts, and cannot perform real transactions.

## Current deployment status

Static authentication, the learned domain/intent router, OOD refusal, and the
synthetic dashboard are live. The public ChatInterface and isolated two-pass
model service are verified on an RTX PRO 6000 Blackwell ZeroGPU partition,
including model-selected reads and writes for both authenticated demo users.

## Live artifacts

- Live application: https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc
- Public source: https://github.com/spkc83/retail-bank-servicing-poc
- Model-development source:
  https://github.com/spkc83/retail-bank-servicing
- 9B conversational model:
  https://huggingface.co/spkc83/retail-bank-servicing-moe-9b
- Domain and intent router:
  https://huggingface.co/spkc83/retail-bank-domain-intent-router
- Governed router dataset:
  https://huggingface.co/datasets/spkc83/retail-bank-router-training-data

The application code pins immutable model and router weight revisions for
reproducible inference; the links above intentionally point to the current,
clean public documentation.

## Request path

```text
Static login
  → CPU dual-head domain/intent router
  → intent narrows the tool schema presented to the ZeroGPU 9B model
  → 9B model emits one constrained tool call
  → server validates tool name, arguments, identity scope, and write authorization
  → per-session ephemeral SQLite executes against synthetic records
  → ZeroGPU 9B model receives the tool result and writes the final response
```

OOD and credential-bearing requests are rejected before model inference.
Customer identity is derived only from Gradio authentication. Tool arguments
cannot select a customer. Each browser page session receives an isolated,
TTL-limited database cloned from the immutable synthetic seed. The database
files live only in the Space's temporary runtime storage, allowing the CPU web
process and ZeroGPU worker process to share the same session state.

## Supported tools

- list accounts, balances, cards, transactions, transfers, and service cases;
- freeze or replace a synthetic card;
- dispute a synthetic debit transaction;
- cancel a synthetic transfer when it is pending.

Write tools require explicit customer language such as “freeze,” “replace,”
“dispute,” or “cancel,” even when the model proposes the tool.

If the 9B model emits malformed syntax for an explicit `cancel ... transfer`
request, the learned `cancel_transfer` intent can repair the operation name.
The repair never supplies a customer or transfer identifier; the policy and
authenticated session backend still resolve and validate the pending synthetic
transfer.

## Authentication

The two demo usernames are `alex.demo` and `maya.demo`. Deployment passwords
are not committed; local tests use non-secret test-only values. The Space reads
the live credentials from the write-only `DEMO_AUTH_JSON` secret. Gradio’s
static authentication is appropriate only for a limited POC; it is not a
production identity system.

## Local verification

```bash
python -m pip install -e '.[dev]'
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-chars"}'
export POC_SKIP_MODEL_LOAD=1
export POC_SKIP_ROUTER_LOAD=1
pytest
ruff check .
```

The skip flags are for UI and service-contract tests only. Release verification
also runs authenticated live ZeroGPU checks for model tool selection, exact
grounding, validated database execution, state mutation, model-authored final
responses, multi-turn context, OOD refusal, and credential rejection.
