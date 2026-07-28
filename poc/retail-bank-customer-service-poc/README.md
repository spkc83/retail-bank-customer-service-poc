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
retail bank. A CPU router classifies the request, a deterministic capability
planner selects supported workflows, a session-isolated SQLite backend executes
against synthetic records, and the 9B banking model runs on Hugging Face
ZeroGPU only to write the final grounded customer response.

Everything in the application is synthetic. It has no connection to a real
bank, cannot access real accounts, and cannot perform real transactions.

## Current deployment status

Static authentication, the learned domain/intent router, OOD refusal, and the
synthetic dashboard are live. The public Gradio Blocks application uses a CPU
dispatch event and a separately registered ZeroGPU model event on an RTX PRO
6000 Blackwell partition. It supports read bundles and one-write-at-a-time
synthetic actions for both authenticated demo users.

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
  → CPU /chat dispatch, credential guard, dual-head router, and capability planner
  → direct conversational/policy response, or a unique pending model turn
  → registered ZeroGPU model event for backend-executing workflows only
  → server validates workflow, arguments, identity scope, and write authorization
  → per-session ephemeral SQLite executes against synthetic records
  → 9B finalizer receives sanitized grounded results when a model answer is needed
  → server validates the final response before returning it
```

Credential-bearing requests are rejected before model inference. The
capability planner handles greetings and acknowledgements directly, returns a
stock response for explicit non-banking subjects, and returns an honest
unsupported-banking response when the request is financial-services related but
outside the POC backend. Those direct paths never allocate ZeroGPU. Only a
model-backed pending turn changes the hidden session state that triggers the
registered GPU event. The input and reset controls remain disabled until model
success or failure, and a reset epoch invalidates stale queued turns. Customer
identity is derived only from Gradio authentication, never from pending state.
Tool arguments cannot select a customer. Each browser page session receives an
isolated, TTL-limited database cloned from the immutable synthetic seed. The
database files live only in the Space's temporary runtime storage, allowing
successive ZeroGPU workers to share the same session state.

## Supported workflows

- list accounts, balances, cards, transactions, transfers, and service cases;
- freeze or replace a synthetic card;
- dispute a synthetic debit transaction;
- cancel a synthetic transfer when it is pending.

Read-only requests can execute multiple backend reads in one response, such as
transfers plus recent transactions. The mailing-address history preset is
backed by the limited synthetic service-case records, not a full profile-change
audit table. Server validation uses a labeled deterministic grounded repair for
multi-read answers and for model drafts that omit required balance labels,
the limited-history qualifier, or verified workflow values.

Write workflows require explicit customer language such as “freeze,”
“replace,” “dispute,” or “cancel.” The planner permits only one write per user
turn and does not combine writes with reads. A deterministic,
authenticated-session resolver matches customer-described records, such as card
last four digits, transfer recipient or amount, or latest purchase, to exactly
one synthetic backend record. Unknown, completed, or ambiguous targets fail
safely. Write actions commit only after the ZeroGPU model's final answer passes
credential, unsafe-output, and internal-identifier validation; unavailable or
unsafe finalization rolls back the synthetic action. A contradictory or
incomplete write acknowledgement is replaced with a response rendered only
from the verified action result before commit.

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
also runs authenticated live ZeroGPU checks for deterministic workflow
selection, exact grounding, validated database execution, state mutation,
model-authored final responses, sanitized alternating history, OOD refusal, and
credential rejection.
