# System Overview

This repository builds a model-driven synthetic retail-bank service agent. The
active release has three runtime pieces:

- A CPU dual-head router for domain/OOD gating and intent diagnostics.
- A Granite 8.79B generative agent fine-tuned with PEFT/LoRA.
- A synthetic SQLite banking backend wrapped by the Gradio/ZeroGPU POC.

All customer data is fictional. No file in this repository connects to a real
bank.

## Runtime Flow

```text
Authenticated synthetic customer
  -> CPU dual-head router
  -> high-confidence OOD: governed scope response
  -> in-domain or uncertain: ZeroGPU Granite 8.79B generation
  -> direct answer, clarification, or tagged-JSON tool call
  -> synthetic SQLite tool execution
  -> tool result returned to the model
  -> model-authored final response
```

The implementation is split across these files:

| Step | Code |
| --- | --- |
| Gradio event and OOD shortcut | [../poc/retail-bank-customer-service-poc/app.py](../poc/retail-bank-customer-service-poc/app.py) |
| CPU router loading and prediction | [../poc/retail-bank-customer-service-poc/router.py](../poc/retail-bank-customer-service-poc/router.py) |
| Model/tool loop | [../poc/retail-bank-customer-service-poc/model_service.py](../poc/retail-bank-customer-service-poc/model_service.py) |
| ZeroGPU model loading and deterministic decoding | [../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| Synthetic bank state and tool execution | [../poc/retail-bank-customer-service-poc/mock_bank.py](../poc/retail-bank-customer-service-poc/mock_bank.py) |

## Component Responsibilities

### Dual-head router

The router is a shared-encoder classifier with:

- a binary supported-banking/OOD head;
- a 77-way Banking77 intent head.

The POC loads the artifact from `spkc83/retail-bank-domain-intent-router` at
revision `136ee159d19cda7f585dd122907bbeb1ef4ec4db`. The router uses two
serving thresholds in [../poc/retail-bank-customer-service-poc/router.py](../poc/retail-bank-customer-service-poc/router.py):

- banking probability below `0.165`: high-confidence OOD;
- banking probability at least `0.50`: in-domain;
- the middle region: uncertain.

High-confidence OOD requests receive the static governed response from
[../poc/retail-bank-customer-service-poc/responses.py](../poc/retail-bank-customer-service-poc/responses.py).
Uncertain requests continue to the 8.79B model. The top intent candidates are
diagnostics only; they are not added to the prompt and they do not choose tools.

### Granite generative agent

The POC loads `spkc83/retail-bank-agent-9b` at revision
`085df3d089cfadd77424b548542da0390a54a23e` by default in
[../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py).
The model uses deterministic generation:

- `do_sample=False`;
- max new tokens `512`;
- FP16 weights on CUDA in the live Space.

The model receives the system prompt, retained conversation history, and the
nine public tool schemas. It can answer directly or emit Granite tagged-JSON
tool calls.

### Tool loop

[../poc/retail-bank-customer-service-poc/model_service.py](../poc/retail-bank-customer-service-poc/model_service.py)
owns the model/tool protocol:

1. Build a token-budgeted prompt with an 8,192-token input budget.
2. Ask the model for a first response.
3. Parse `<tool_call>...</tool_call>` blocks when present.
4. Validate tool names, argument types, argument ranges, call IDs, and indexes.
5. Execute each tool against the session-isolated synthetic bank.
6. Append tool results to the conversation.
7. Ask the same model for the grounded final response.
8. Repeat only if the model emits another valid tool call, up to eight total
   tool calls.

The runtime does not infer intent, repair malformed calls, rename tools, fill
missing arguments, or synthesize a banking answer if the model fails.

### Synthetic bank backend

[../poc/retail-bank-customer-service-poc/mock_bank.py](../poc/retail-bank-customer-service-poc/mock_bank.py)
creates a SQLite database per authenticated user/session pair. It supports:

- read tools: `list_accounts`, `list_cards`, `list_service_cases`,
  `list_transactions`, `list_transfers`;
- write tools: `cancel_transfer`, `dispute_transaction`, `freeze_card`,
  `replace_card`.

The seed records live in
[../poc/retail-bank-customer-service-poc/synthetic_bank.json](../poc/retail-bank-customer-service-poc/synthetic_bank.json).
Each browser session gets isolated state, so a write action in one demo session
does not modify another session.

## Training-Time Counterparts

Runtime behavior mirrors the SFT and evaluation code:

| Runtime behavior | Training/evaluation counterpart |
| --- | --- |
| Public tool schema | `public_tool_manifest()` in [../src/hello_slm/banking_tool_sft_data.py](../src/hello_slm/banking_tool_sft_data.py) |
| Granite tagged-JSON parsing | [../src/hello_slm/banking_tool_wire.py](../src/hello_slm/banking_tool_wire.py) |
| Assistant-only targets | `ToolWireAdapter.render_training()` in [../src/hello_slm/banking_tool_wire.py](../src/hello_slm/banking_tool_wire.py) |
| Frozen tool/final-response scoring | [../src/hello_slm/banking_tool_eval.py](../src/hello_slm/banking_tool_eval.py) |
| Router architecture | [../src/hello_slm/banking_dual_head_router.py](../src/hello_slm/banking_dual_head_router.py) |

## Failure Behavior

The POC is explicit about failure boundaries:

- If the router artifact is unavailable or classification fails, the route is
  marked uncertain and the 8.79B model receives the turn.
- If ZeroGPU allocation or generation fails, the UI reports model
  unavailability.
- If the model emits malformed tool syntax or invalid tool arguments, the UI
  reports a model failure and exposes diagnostics.
- The application does not substitute a CPU-authored banking answer for a
  failed model response.

See [../poc/retail-bank-customer-service-poc/README.md](../poc/retail-bank-customer-service-poc/README.md)
for the public POC card and live verification expectations.
