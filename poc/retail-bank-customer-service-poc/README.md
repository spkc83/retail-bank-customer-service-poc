---
title: Retail Bank Customer Service POC
emoji: 🏦
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
python_version: 3.12
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

## Live artifacts

- Live application: https://huggingface.co/spaces/spkc83/retail-bank-customer-service-poc
- Public source: https://github.com/spkc83/retail-bank-customer-service-poc
- 9B conversational model:
  https://huggingface.co/spkc83/retail-bank-servicing-moe-9b/tree/b2466ca4b157f420432a5e20a14573e83954deae
- Domain and intent router:
  https://huggingface.co/spkc83/retail-bank-domain-intent-router/tree/e7d928e5cf8c8be0883625f276c4e6c85c35eaf1
- Governed router dataset:
  https://huggingface.co/datasets/spkc83/retail-bank-router-training-data

## Request path

```text
Static login
  → CPU dual-head domain/intent router
  → ZeroGPU 9B model emits one constrained tool call
  → server validates tool name, arguments, identity scope, and write authorization
  → per-session in-memory SQLite executes against synthetic records
  → ZeroGPU 9B model receives the tool result and writes the final response
```

OOD and credential-bearing requests are rejected before GPU allocation.
Customer identity is derived only from Gradio authentication. Tool arguments
cannot select a customer. Each browser page session receives an isolated,
TTL-limited in-memory database cloned from the immutable synthetic seed.

## Supported tools

- list accounts, balances, cards, transactions, transfers, and service cases;
- freeze or replace a synthetic card;
- dispute a synthetic debit transaction;
- cancel a synthetic transfer when it is pending.

Write tools require explicit customer language such as “freeze,” “replace,”
“dispute,” or “cancel,” even when the model proposes the tool.

## Authentication

The two demo usernames are `alex.demo` and `maya.demo`. Passwords are not
committed. The Space reads them from the write-only `DEMO_AUTH_JSON` secret.
Gradio’s static authentication is appropriate only for a limited POC; it is not
a production identity system.

## Local verification

```bash
python -m pip install -e '.[dev]'
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-chars"}'
export POC_SKIP_MODEL_LOAD=1
export POC_SKIP_ROUTER_LOAD=1
pytest
ruff check .
```

The skip flags are for UI and service-contract tests only. A release requires a
live ZeroGPU test proving model tool selection, validated database execution,
state mutation, and model-authored final response for both demo accounts.
