# Retail Bank Agent

Training, evaluation, and public demonstration code for a model-driven
retail-bank customer-service agent.

The active generative model is a merged BF16 LoRA adaptation of
`ibm-granite/granite-4.1-8b` at a pinned base revision. It has 8.79 billion
parameters, uses Granite's native tagged-JSON tool-call format, and is trained
on 5,000 governed synthetic conversations. The earlier custom Qwen2-MoE model
is retained only as an evaluation control.

## Public artifacts

- Model: https://huggingface.co/spkc83/retail-bank-agent-9b
- Tool-use SFT dataset:
  https://huggingface.co/datasets/spkc83/retail-bank-agent-sft
- Domain and intent router:
  https://huggingface.co/spkc83/retail-bank-domain-intent-router
- Router training dataset:
  https://huggingface.co/datasets/spkc83/retail-bank-router-training-data
- Public ZeroGPU application:
  https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc
- Standalone application source:
  https://github.com/spkc83/retail-bank-servicing-poc
- Model-development source:
  https://github.com/spkc83/retail-bank-servicing

## Runtime

```text
Authenticated synthetic customer
  → CPU dual-head classifier identifies OOD and predicts advisory intents
  → in-domain or uncertain request enters the ZeroGPU 8.79B agent
  → model responds directly or emits tagged-JSON tool calls
  → generated calls execute against session-isolated synthetic SQLite
  → tool results return to the model for a grounded final response
```

The classifier does not select tools. The generative model owns the
conversation, clarification, tool choice, public arguments, and final wording.
The runtime performs mechanical schema validation, mock-tool execution, context
budgeting, and diagnostics. High-confidence OOD requests receive the governed
financial-services scope response without invoking the 8.79B model.

This is a research demonstration. It has no connection to a bank and cannot
access or modify real accounts.

## Model and training

- Base: `ibm-granite/granite-4.1-8b`
- Base revision:
  `1504002f650e656a0a3789d99574df12e3e94ed0`
- Architecture: dense decoder-only causal transformer
- Parameters: 8,791,592,960
- Adaptation: BF16 LoRA over attention and MLP projections
- Tool wire: native tagged JSON
- Maximum training sequence: 2,048 tokens
- Deployment artifact: merged BF16 checkpoint plus a separate adapter copy

The governed corpus contains 3,502 training, 748 validation, and 750 frozen
test conversations. It covers all nine mock-bank tools, tool errors,
clarification, banking FAQ, hard-negative private-field requests, OOD,
multi-turn context, and ordered multi-tool calls. Every tool trajectory is
replayed against deterministic synthetic state before inclusion.

All generative rows are self-authored synthetic data under MIT. Banking77 and
CLINC remain classifier/evaluation-only. The quarantined Bitext corpus
contributes no rows to this release.

See the [tool-use dataset card](data_cards/retail-bank-agent-sft.md) for split,
coverage, provenance, and privacy details.

Generate and validate the corpus:

```bash
PYTHONPATH=src python scripts/banking_v2/prepare_tool_sft_data.py \
  --output-dir data/banking-v3-tool-sft \
  --pilot-count 5000
```

Inspect the guarded training plan without starting remote work:

```bash
PYTHONPATH=src python scripts/banking_v2/cloud_train_tool_sft.py \
  --manifest data/banking-v3-tool-sft/manifest.json
```

The full Hugging Face Jobs entry point is
`scripts/banking_v2/hf_job_tool_sft.py`. It requires exact source, dataset, and
base revisions; a five-hour outer timeout; a four-hour optimizer callback; and
fresh-base adapter merge/reload parity before Hub upload.

Launch through the durable-storage wrapper:

```bash
scripts/banking_v2/run_remote_training_job.sh \
  "$(git rev-parse HEAD)" \
  c0e0be08f9d56f382e3c85a6bca1e4f4090eacac
```

The complete implementation and acceptance contract is
[the banking v3 specification](specs/banking-v3/01-tool-use-sft-plan.md).

After the merged checkpoint is published, run the frozen 750-record,
two-phase tool/final-response evaluation with exact revisions:

```bash
bash scripts/banking_v2/run_remote_tool_eval_job.sh \
  "$(git rev-parse HEAD)" \
  MODEL_REVISION \
  c0e0be08f9d56f382e3c85a6bca1e4f4090eacac
```

The evaluation job performs deterministic decoding only. It executes no tools
and applies no output repair; grounded-final scoring uses the dataset's
replay-validated canonical tool results. Predictions, metadata, and the scored
report are persisted to the mounted bucket and published under the model
repository's `evaluation/` directory.

The earlier dense-to-MoE design remains documented only as the control
architecture in
[the historical routing note](specs/banking-v2/04-dense-to-moe-routing.md).

## Dual-head classifier

The CPU classifier shares a DistilBERT encoder between:

- a binary supported-banking/OOD head;
- a 77-way Banking77 intent head.

Its intent predictions are advisory model context, not orchestration commands.
The released artifact reports intent macro F1 `0.948425`, in-domain
false-refusal rate `0.005099`, and OOD false-accept rate `0.020109`. The
calibrated lower boundary is `0.165`; the POC treats scores from `0.165` to
`0.50` as uncertain and asks the 9B model to adjudicate them.

## Public POC

The Gradio application under `poc/retail-bank-customer-service-poc/` includes:

- two static demonstration identities selected through Space authentication;
- the CPU dual-head classifier;
- an 8,192-token, complete-interaction conversation budget;
- model-authored direct answers, clarification, tools, and grounded finals;
- a labeled model reflection pass when a base draft emits no tool call;
- session-isolated synthetic SQLite state;
- diagnostics with the exact model ID, revision, runtime device, raw model
  passes, generated calls, results, and prompt/output hashes;
- preset read, write, multi-turn, FAQ, and OOD scenarios.

ZeroGPU owns the model generation boundary. If GPU inference fails, the
application reports the failure and does not synthesize a Python banking
answer.

## Verification

```bash
python -m pytest -q tests
ruff check .
MYPYPATH=src mypy src scripts tests
uv lock --check
```

POC verification also requires its application dependencies and a live
ZeroGPU round trip:

```bash
POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
  pytest -q poc/retail-bank-customer-service-poc/tests
```

## Repository map

```text
configs/        pinned model candidates and training configurations
data_cards/     classifier-dataset documentation
data/sources/   governed source locks; generated data is ignored
model_cards/    released model and classifier documentation
poc/            authenticated Gradio/ZeroGPU application
scripts/        corpus, training, evaluation, and Hub job entry points
specs/          data, architecture, evaluation, and serving contracts
src/hello_slm/  dataset, tool-wire, evaluator, router, and local UI modules
tests/          banking regression tests
```

## Safety and license

Do not enter passwords, PINs, one-time codes, full account numbers, or
payment-card details. The model may produce incorrect or inconsistent banking
guidance. All operations affect only isolated synthetic state.

Repository code and the released synthetic tool-use dataset are MIT licensed.
Upstream base-model and classifier artifacts retain their own licenses.
