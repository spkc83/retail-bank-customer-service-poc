# Inference and ZeroGPU POC

This page explains the active authenticated Gradio POC in
[`poc/retail-bank-customer-service-poc/`](../poc/retail-bank-customer-service-poc/).
It documents the current Granite 9B model path, the CPU dual-head router, the
model-owned tool loop, session state, diagnostics, local tests, and the Space
deployment surface.

Everything in the POC is synthetic. It has no connection to a real bank, cannot
access real accounts, and must not receive credentials, full account numbers,
payment-card details, or real customer data.

## Active Artifacts

The POC uses the released artifacts below. Do not substitute branch names such
as `main` where these revisions are required.

| Role | Repository | Immutable revision | Evidence |
| --- | --- | --- | --- |
| Generative agent | `spkc83/retail-bank-agent-9b` | `085df3d089cfadd77424b548542da0390a54a23e` | [`zero_gpu_runtime.py`](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py), [`model card`](../model_cards/retail-bank-agent-9b.md) |
| Agent base | `ibm-granite/granite-4.1-8b` | `1504002f650e656a0a3789d99574df12e3e94ed0` | [`configs/banking-tool-sft-granite.toml`](../configs/banking-tool-sft-granite.toml), [`model card`](../model_cards/retail-bank-agent-9b.md) |
| Tool-use SFT dataset | `spkc83/retail-bank-agent-sft` | `183e7e1ed1aba9c3d7155e7b83b64dc854935055` | [`data card`](../data_cards/retail-bank-agent-sft.md), [`model card`](../model_cards/retail-bank-agent-9b.md) |
| Dual-head router | `spkc83/retail-bank-domain-intent-router` | `136ee159d19cda7f585dd122907bbeb1ef4ec4db` | [`router.py`](../poc/retail-bank-customer-service-poc/router.py), [`router card`](../model_cards/retail-bank-domain-intent-router.md) |
| Router dataset | `spkc83/retail-bank-router-training-data` | `54ff186a03501d76dc643dbed3d82729267ce811` | [`train_dual_head_router.py`](../scripts/banking_v2/train_dual_head_router.py), [`router card`](../model_cards/retail-bank-domain-intent-router.md) |

For the complete artifact ledger, see
[`docs/reference/artifacts.md`](reference/artifacts.md).

## Runtime Flow

The public app is [`app.py`](../poc/retail-bank-customer-service-poc/app.py).
At a high level, one authenticated chat turn follows this path:

```text
Gradio authenticated user
  -> CPU dual-head router
  -> high-confidence OOD returns the governed stock response
  -> in-domain or uncertain turn enters one ZeroGPU event
  -> Granite 9B either answers directly or emits tagged-JSON tool calls
  -> generated calls execute against session-isolated SQLite
  -> tool results return to Granite 9B
  -> Granite 9B writes the final customer-facing response
```

The router does not choose tools, write arguments, add facts to the prompt, or
repair model output. The runtime only budgets context, validates generated tool
syntax, executes the synthetic backend, and records diagnostics.

## Static Authentication

Authentication is loaded by
[`auth.py`](../poc/retail-bank-customer-service-poc/auth.py). The app accepts
exactly two usernames:

- `alex.demo`
- `maya.demo`

Passwords come from the `DEMO_AUTH_JSON` environment variable. The value must be
a JSON object with exactly those two usernames, different passwords, and each
password must contain at least 12 characters.

Example local value:

```bash
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-more"}'
```

The auth layer only selects one of the two synthetic customer records and the
Gradio session hash. It is not a bank identity provider.

## Router Loading and Thresholds

The router code lives in
[`router.py`](../poc/retail-bank-customer-service-poc/router.py). Startup calls:

```python
LearnedBankingRouter.from_hub()
```

That method downloads `spkc83/retail-bank-domain-intent-router` at revision
`136ee159d19cda7f585dd122907bbeb1ef4ec4db`, verifies
`manifest.json`, loads the tokenizer and DistilBERT encoder with
`trust_remote_code=False`, and loads `classifier_heads.safetensors`.

The released artifact's calibrated lower boundary is `0.165`. Serving uses two
boundaries:

- banking probability `< 0.165`: `out_of_domain`
- banking probability `>= 0.50`: `in_domain`
- banking probability from `0.165` through `< 0.50`: `uncertain`

Uncertain turns continue to the Granite 9B model. If the router is unavailable
or classification fails after startup, [`app.py`](../poc/retail-bank-customer-service-poc/app.py)
records an uncertain route and delegates the turn to the model.

The top three Banking77 intents are diagnostics only.

## Model Loading

ZeroGPU model loading is isolated in
[`zero_gpu_runtime.py`](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py).
By default it loads:

- model ID: `spkc83/retail-bank-agent-9b`
- model revision: `085df3d089cfadd77424b548542da0390a54a23e`
- dtype: `torch.float16`
- device: CUDA
- generation: deterministic, `do_sample=False`

The model ID and revision can be overridden with:

```bash
export RETAIL_BANK_MODEL_ID=spkc83/retail-bank-agent-9b
export RETAIL_BANK_MODEL_REVISION=085df3d089cfadd77424b548542da0390a54a23e
```

For local tests that should not load the 9B model, set:

```bash
export POC_SKIP_MODEL_LOAD=1
```

When `POC_SKIP_MODEL_LOAD=1`, the module installs a local `spaces.GPU`
decorator stub and leaves the tokenizer/model unset. Tests can then validate
routing, auth, state, parsing, and UI plumbing without downloading the model.

## ZeroGPU Boundary

The model turn is registered in
[`app.py`](../poc/retail-bank-customer-service-poc/app.py) with:

```python
@spaces.GPU(size="large", duration=90)
def run_model_turn(...):
    ...
```

The whole route-plus-generation turn is queued as a Gradio event. If ZeroGPU
allocation or model generation fails, the app returns
[`MODEL_FAILURE_RESPONSE`](../poc/retail-bank-customer-service-poc/responses.py).
It does not synthesize a Python-authored banking answer.

## Tool Loop

The agent loop is implemented in
[`model_service.py`](../poc/retail-bank-customer-service-poc/model_service.py).
The model receives one system message plus the public tool manifest. The system
message instructs the already-authenticated synthetic-bank agent to use tools
for customer-specific records or actions and never ask for private IDs,
passwords, PINs, or credentials.

The active public tools are:

| Tool | Purpose | Arguments |
| --- | --- | --- |
| `list_accounts` | List accounts and balances | none |
| `list_cards` | List cards and statuses | none |
| `list_service_cases` | List recent service cases | none |
| `list_transactions` | List recent account transactions | optional `limit` from 1 to 20 |
| `list_transfers` | List transfers and statuses | none |
| `freeze_card` | Freeze a card | optional `last4` |
| `replace_card` | Request card replacement | optional `last4` |
| `dispute_transaction` | Dispute one transaction | optional `description` |
| `cancel_transfer` | Cancel one pending transfer | optional `recipient` |

Granite emits tagged JSON:

```text
<tool_call>{"name":"list_accounts","arguments":{}}</tool_call>
```

The runtime parses the tag, validates the JSON object, checks the tool name and
argument schema, executes the call, appends a correlated tool result, and asks
the same model for the next response. The model may emit another tool call after
seeing a tool result. The loop stops when the model emits a normal assistant
response.

Limits:

- maximum input budget: 8,192 tokens
- maximum generation per pass: 512 tokens
- maximum tool calls per turn: 8

Unsupported tool names, duplicate call IDs, out-of-order indexes, malformed
JSON, and invalid argument types raise protocol errors. Backend errors are
returned to the model as safe tool-result envelopes.

## Conversation Budget

The budgeter is
[`select_token_budgeted_context`](../poc/retail-bank-customer-service-poc/model_service.py).
It keeps complete interaction groups:

- user message
- assistant tool call message, when present
- correlated tool result messages, when present
- final assistant response

The latest current interaction and system message are retained first. Then the
newest complete prior groups are added while the rendered prompt and tool
definitions fit within 8,192 input tokens. A tool chain is never split across
the context boundary.

The app reserves 512 new tokens for each model pass.

## Synthetic SQLite State

State setup is in
[`state.py`](../poc/retail-bank-customer-service-poc/state.py). The seed data is
[`synthetic_bank.json`](../poc/retail-bank-customer-service-poc/synthetic_bank.json).

`SessionBankRegistry` in
[`mock_bank.py`](../poc/retail-bank-customer-service-poc/mock_bank.py) creates
one SQLite database per `(username, Gradio session hash)` pair. By default,
files are written under:

```text
/tmp/retail-bank-servicing-poc
```

Override the directory with:

```bash
export POC_SESSION_DB_DIR=/tmp/my-retail-bank-poc
```

Session behavior:

- each session starts from deterministic synthetic JSON records;
- sessions expire after 7,200 seconds;
- at most 32 sessions are retained;
- write tools update only the session database;
- reset reseeds the current user's session database.

The sidebar snapshot is rendered from the current session database, so write
tools such as `freeze_card` and `cancel_transfer` are visible immediately after
the model turn completes.

## Diagnostics

Diagnostics are rendered by
[`_render_diagnostics`](../poc/retail-bank-customer-service-poc/app.py). They
are part of the proof that a turn used the active model path.

The panel shows:

- route: `in_domain`, `uncertain`, or `out_of_domain`
- in-domain and OOD probabilities
- whether conversation context changed the route
- response path, such as `direct_answer`, `base_tool`, or `base_tool_chain`
- top intent candidates
- generated tool calls and public arguments
- tool-result success or safe error code
- model pass labels, input-token counts, prompt SHA-256 values, raw output
  SHA-256 values, raw outputs, runtime device, and CUDA device name
- generation call count
- model ID and exact model revision
- `SPACE_COMMIT_SHA`, when provided by the Space runtime
- visible response SHA-256

A successful live model turn should show:

- `Model: spkc83/retail-bank-agent-9b`
- `Exact model revision: 085df3d089cfadd77424b548542da0390a54a23e`
- `Registered execution boundary: ZeroGPU large`
- a CUDA runtime device for model passes

High-confidence OOD diagnostics intentionally show no model passes.

## Local Tests

Run repository tests from the repo root:

```bash
python -m pytest -q tests
```

Run the POC tests without loading the 9B model or router artifact:

```bash
cd poc/retail-bank-customer-service-poc
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-more"}'
export POC_SKIP_MODEL_LOAD=1
export POC_SKIP_ROUTER_LOAD=1
python -m pytest -q tests
```

Run static checks from the repo root:

```bash
ruff check .
MYPYPATH=src mypy src scripts tests
uv lock --check
```

Local POC tests validate auth, router behavior, model-service protocol handling,
SQLite state, Gradio app behavior, and the ZeroGPU test stub. They do not prove
a live GPU generation unless the skip variables are removed in a Space or other
CUDA-capable environment.

## Space Deployment Surface

The Space app files are in
[`poc/retail-bank-customer-service-poc/`](../poc/retail-bank-customer-service-poc/):

- [`README.md`](../poc/retail-bank-customer-service-poc/README.md): Space card
  front matter and public operating notes
- [`app.py`](../poc/retail-bank-customer-service-poc/app.py): Gradio app
- [`zero_gpu_runtime.py`](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py):
  model load and deterministic generation
- [`requirements.txt`](../poc/retail-bank-customer-service-poc/requirements.txt):
  Space dependencies
- [`auth.py`](../poc/retail-bank-customer-service-poc/auth.py): demo auth
- [`router.py`](../poc/retail-bank-customer-service-poc/router.py): CPU router
- [`model_service.py`](../poc/retail-bank-customer-service-poc/model_service.py):
  model-owned tool loop
- [`mock_bank.py`](../poc/retail-bank-customer-service-poc/mock_bank.py): SQLite
  synthetic backend
- [`state.py`](../poc/retail-bank-customer-service-poc/state.py): session
  registry setup
- [`synthetic_bank.json`](../poc/retail-bank-customer-service-poc/synthetic_bank.json):
  seed data

Deploying to the public Space is an external production action. Do not run a
deployment command without explicit authorization for that deployment. Before a
deployment, verify the local POC tests and set the Space secret `DEMO_AUTH_JSON`
to the exact two demo users.

The active public Space is:

```text
https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc
```

After deployment, run live read, write, multi-tool, clarification, FAQ, OOD, and
multi-turn cases. The diagnostics panel must show the active model revision and
CUDA-backed generation for model-handled turns.
