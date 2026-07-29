# Retail Bank Servicing Model Development

Training, evaluation, routing, and demonstration code for a focused retail-bank
customer-service model. The repository contains only the banking model-development
track and its public proof-of-concept application.

The released language model is an 8.94B-parameter Qwen2-MoE checkpoint derived
from the pinned `Qwen/Qwen2.5-1.5B-Instruct` model. It is domain-adapted, not
trained from random initialization. Approximately 2.07B parameters are active
per token.

## Public artifacts

- Model: https://huggingface.co/spkc83/retail-bank-servicing-moe-9b
- Domain and intent router:
  https://huggingface.co/spkc83/retail-bank-domain-intent-router
- Router training dataset:
  https://huggingface.co/datasets/spkc83/retail-bank-router-training-data
- Public application:
  https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc
- Standalone application source:
  https://github.com/spkc83/retail-bank-servicing-poc
- Model-development source:
  https://github.com/spkc83/retail-bank-servicing

## System

```text
Authenticated request
  → credential guard
  → CPU-resident dual-head router gates OOD requests
  → intent probabilities guide the ZeroGPU 9B MoE agent
  → 9B model responds directly or emits Qwen tool calls
  → a no-tool draft receives a labeled 9B tool-use reflection pass
  → session-isolated synthetic SQLite executes generated calls
  → tool results return to the 9B model for its final response
```

The POC is a behavioral experiment. The dual-head router exposes a three-way
domain decision plus its top three intent predictions. Only OOD
bypasses the 9B model. For allowed and uncertain turns, the 9B model owns
conversation, clarification, tool selection, tool arguments, and final wording.
The runtime performs mechanical parsing and direct mock-tool execution; it does
not replace the model's answer with a deterministic banking response.
Short follow-ups may be classified with the immediately preceding banking
exchange. The reflection pass can emit a valid tool call or explicitly retain
the untouched base answer; it does not map classifier intents to tools.

The application is synthetic. It has no connection to a bank and cannot access
or modify real accounts.

## Model architecture

- `Qwen2MoeForCausalLM`
- 8,943,713,792 total parameters
- approximately 2,073,443,840 active parameters per token
- 28 layers
- 28 routed experts per layer, top-2 routing
- BF16 released checkpoint
- inherited 151,936-token Qwen vocabulary

Compatible embeddings, attention, normalization, and language representations
were copied from the pinned 1.5B base model. The dense MLP was expanded into a
shared expert and zero-initialized routed residual experts. Router weights and
routed down projections were then adapted for 1,000 optimizer steps.

See [the architecture specification](specs/banking-v2/04-dense-to-moe-routing.md)
for the conversion and routing details.

## Training data

Generative adaptation uses the prepared Bitext retail-banking corpus plus a
small self-authored set of governed out-of-domain and multi-turn conversations.
The released split contains 22,033 training, 1,008 validation, and 1,001 test
conversations.

Banking77 and CLINC150 are classifier-only sources. They are used for the
dual-head router and are prohibited from the generative training lane.

Data preparation is deterministic and enforces:

- pinned source revisions and fingerprints;
- placeholder normalization;
- PII-like value scrubbing;
- exact and clustered cross-split deduplication;
- explicit source licenses and trainability;
- exact stock responses only for governed OOD records.

```bash
python -m pip install -e '.[dev]'
python -m hello_slm.banking_data audit-sources
python -m hello_slm.banking_data prepare
```

Generated corpora and downloaded source snapshots are ignored by Git. Their
tracked lock files are under `data/sources/`.

## MoE training and evaluation

The reproducible model shape and run limits are in
`configs/banking-v2-moe-9b.toml`. The executable worker is
`scripts/banking_v2/cloud_train_banking_moe.py`.

Run the offline tiny architecture test:

```bash
PYTHONPATH=src python scripts/banking_v2/cloud_train_banking_moe.py \
  --run-tiny-smoke \
  --max-steps 1 \
  --output-dir artifacts/banking-v2-tiny-smoke
```

Inspect the guarded full-run plan without launching paid infrastructure:

```bash
PYTHONPATH=src python scripts/banking_v2/train_banking_moe.py
```

Remote execution requires both the worker flag and its explicit confirmation
environment variable. Checkpoint resumes verify the base revision, dataset
fingerprint, converted-state manifest, optimizer, scheduler, and RNG state.

The released run completed 1,000 steps on an RTX PRO 6000:

- final training loss: `1.2638`;
- validation loss: `0.7775`;
- every expert-health gate passed at steps 250, 500, 750, and 1,000.

These values demonstrate a completed run, not production readiness. The raw
model still requires external domain, credential, tool, and output guards.

## Dual-head router

The CPU router shares a DistilBERT encoder between:

- a binary supported-banking/OOD head;
- a 77-way Banking77 intent head.

Prepare and train it locally:

```bash
PYTHONPATH=src python scripts/banking_v2/prepare_dual_head_router_data.py
PYTHONPATH=src python scripts/banking_v2/train_dual_head_router.py --help
```

The public router artifact reports intent macro F1 `0.951208`, in-domain
false-refusal rate `0.013689`, and OOD false-accept rate `0.007733` at a
calibrated banking threshold of `0.98`.

## Local model lab

The Shiny application loads a local Transformers checkpoint and supports
multi-turn banking chat:

```bash
RETAIL_BANK_MODEL=/path/to/checkpoint \
  shiny run --reload src/hello_slm/banking_shiny_app.py
```

The canonical BF16 checkpoint needs more than 12 GB of VRAM. The conversion
script creates a separate GGUF copy and quantizes it to `Q4_K_M`; it does not
modify the released BF16 model. The resulting file is about 5.1 GiB and fits in
the 12 GB TITAN V, but local inference is not a supported release path yet:
current llama.cpp builds do not load this tied-output Qwen2-MoE checkpoint
without a loader workaround, and the tested workaround did not produce valid
text. Use the public ZeroGPU application for validated model inference.

With the model snapshot and llama.cpp checked out locally:

```bash
scripts/banking_v2/quantize_local_gguf.sh \
  /path/to/retail-bank-servicing-moe-9b \
  /path/to/llama.cpp \
  artifacts/gguf

sha256sum artifacts/gguf/retail-bank-servicing-moe-9b-q4_k_m.gguf
```

## Public POC

The deployable Gradio source is in
`poc/retail-bank-customer-service-poc/`. It includes:

- two static demonstration accounts configured through a Space secret;
- a learned CPU domain/intent router with OOD gating and
  top-three intent guidance;
- CPU session-isolated synthetic SQLite state;
- ZeroGPU 9B-owned conversation and Qwen tool calling;
- an experimental labeled 9B reflection pass after a no-tool base draft;
- an 8,192-token input budget retaining complete user/assistant/tool
  interactions without splitting a turn;
- diagnostics for route probabilities, intent candidates, generated tool calls,
  tool results, response path, and per-generation prompt/output hashes;
- preset read, write, multi-turn, sensitive-data, and OOD cases.

The hidden conversation state stores complete user, assistant, tool-call, and
tool-result messages. Each inference builds a tokenizer-measured context from
the newest complete interactions up to 8,192 input tokens and reserves 512
tokens for generation. The current synthetic data has limited address-history
coverage through service-case records; it is not a full customer-profile audit
log.

ZeroGPU compatibility requires the complete chat turn to be registered directly
in the Gradio event graph. Every submitted turn enters that one managed event;
the CPU-resident dual-head router runs inside its worker, OOD
returns the stock response without invoking the 9B generator, and all other
turns continue to model inference. If ZeroGPU fails, the UI reports model
unavailability and does not synthesize a banking response on CPU.

## Verification

```bash
python -m pytest \
  tests/test_banking_chat_runtime.py \
  tests/test_banking_cloud_worker.py \
  tests/test_banking_data.py \
  tests/test_banking_dual_head_router.py \
  tests/test_banking_hf_generator.py \
  tests/test_banking_moe.py \
  tests/test_banking_policy.py \
  tests/test_banking_router_data.py \
  tests/test_banking_router_training.py \
  tests/test_banking_shiny_app.py \
  poc/retail-bank-customer-service-poc/tests
ruff check src scripts tests poc/retail-bank-customer-service-poc
mypy src scripts tests
```

## Repository map

```text
configs/        banking dense-baseline and MoE run configurations
data_cards/     released classifier-dataset documentation
data/sources/   governed source locks; generated data is ignored
model_cards/    released generative-model and router documentation
poc/            current authenticated Gradio application
scripts/        cloud training, evaluation, and router tooling
specs/          banking data, model, routing, evaluation, and serving contracts
src/hello_slm/  banking data, model, router, policy, and local UI modules
tests/          banking-only regression tests
```

## Safety boundary

This is a research demonstration, not financial advice or a production banking
system. Do not enter passwords, PINs, one-time codes, full account numbers, or
payment-card details. The model may produce incorrect, inconsistent, or unsafe
financial guidance. Generated operations run only against isolated synthetic
state so conversational and tool-use behavior can be observed directly.

## License

Repository code is MIT licensed. Dataset rows retain their source licenses:
Bitext generative records are CDLA-Sharing-1.0, self-authored records are MIT,
and Banking77/CLINC150 classifier records are CC-BY-4.0.
