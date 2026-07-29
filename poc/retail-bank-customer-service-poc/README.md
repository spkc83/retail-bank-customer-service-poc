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
short_description: Dual-head router and 9B synthetic banking chat.
---

# Retail Bank Customer Service POC

This authenticated POC tests whether a dual-head OOD/intent classifier plus the
9B banking MoE can provide natural multi-turn customer service and operate a
synthetic retail-bank backend. It is deliberately model-driven rather than a
deterministic servicing simulator.

Everything is fictional. The application has no connection to a bank and
cannot access real accounts or perform real transactions.

## Live artifacts

- Application: https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc
- POC source: https://github.com/spkc83/retail-bank-servicing-poc
- Model-development source: https://github.com/spkc83/retail-bank-servicing
- 9B model: https://huggingface.co/spkc83/retail-bank-servicing-moe-9b
- Dual-head router:
  https://huggingface.co/spkc83/retail-bank-domain-intent-router

## Runtime

```text
Authenticated user and stored session transcript
  → one directly registered ZeroGPU chat event
  → CPU-resident dual-head classifier inside the managed worker
  → OOD: stock response without 9B generation
  → allowed or uncertain: 9B generation
  → generated Qwen <tool_call> JSON, or labeled 9B reflection
  → reflection emits a tool call or retains the untouched base draft
  → direct execution against session-isolated synthetic SQLite
  → tool results appended to model history
  → second 9B generation produces the final answer
```

The domain head has three operating regions: `in_domain`, `uncertain`, and
`out_of_domain`. Every turn enters the managed GPU event so ZeroGPU owns the
complete execution boundary. High-confidence OOD bypasses 9B generation after
classification. The intent head's top three predictions and probabilities are
advisory context for the 9B model; they do not select a tool.
An isolated short or referential reply is reclassified with only the immediately
preceding exchange when that exchange was not OOD. If context still cannot
establish the domain, the turn enters the uncertain path for 9B adjudication.

The 9B model owns greetings, conversational responses, contextual reasoning,
clarification, tool selection, tool arguments, and final wording. The runtime
only parses Qwen's tool-call format and invokes the named mock function. There
is no deterministic workflow planner, authorization policy, grounded-response
repair, or template-generated read fallback.

When the base generation contains no tool call, the same 9B checkpoint receives
a separate tool-use review prompt. It must emit a valid Qwen tool call or
`<use_original/>`. Invalid review output also leaves the base draft untouched.
This temporary test-time-scaling experiment is labeled separately so its
success cannot be mistaken for base-checkpoint tool-call success.

## Conversation context

The application stores the complete valid session transcript, including user
and assistant messages, assistant tool calls, ordered tool results, and final
model responses.

Each inference builds a tokenizer-measured prompt with an 8,192-token input
budget and reserves 512 tokens for generation. The system prompt and complete
current interaction are always retained. Newest prior interactions are added
while they fit; a user/tool-call/tool-result/final-answer chain is never split.
An oversized current interaction fails visibly instead of being truncated.

## Synthetic tools

- list accounts, cards, transactions, transfers, and service cases;
- freeze or replace a card;
- dispute a transaction by merchant description;
- cancel a pending transfer by recipient.

The model may emit multiple calls in one first-pass generation. Calls execute
in generated order. Backend or schema errors return to the 9B model as tool
results so it can explain the outcome naturally.

Because this is an experimental synthetic backend rather than a production
security architecture, generated mock writes execute directly. If a write
succeeds and the second model generation later fails, the dashboard still shows
that synthetic mutation.

## Failure behavior and diagnostics

If ZeroGPU allocation or generation fails, the chat reports that the 9B model
is unavailable. It does not substitute a Python-generated banking answer.

The diagnostics panel exposes domain probabilities, top intent predictions,
generated tool names and arguments, tool status, response path, exact model
revision, and separate prompt/output hashes for the `base`, `reflection`, and
`grounded_final` model calls. Expandable raw outputs, the generation-call count,
and actual runtime/CUDA device metadata make the reflection behavior directly
inspectable. Presets are evaluation prompts, not production routing rules or
proof of generalization.

## Authentication

The demo usernames are `alex.demo` and `maya.demo`. Passwords are supplied
through the Space's write-only `DEMO_AUTH_JSON` secret and displayed on the
login screen for public testing; they are not committed. Authentication exists
only to select isolated synthetic customer records.

## Local verification

```bash
python -m pip install -e '.[dev]'
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-chars"}'
export POC_SKIP_MODEL_LOAD=1
export POC_SKIP_ROUTER_LOAD=1
pytest
ruff check .
```

Release verification additionally requires a live ZeroGPU model-authored tool
round trip; an honest infrastructure failure is not counted as model inference.
