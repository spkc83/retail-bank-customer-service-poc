# Retail Bank Agent Developer Docs

These docs explain the active Granite PEFT retail-bank agent repository for a
junior developer. They describe the current code, cards, scripts, and release
artifacts only.

## What This Repository Builds

The repository builds a synthetic retail-bank customer-service demonstration
with two model components:

- An 8.79B parameter Granite generative agent fine-tuned with PEFT/LoRA for
  conversational tool use.
- A CPU dual-head classifier that detects supported banking requests and emits
  Banking77 intent diagnostics.

The generative agent owns normal conversation, clarification, tool selection,
public tool arguments, and final response wording. The classifier does not
select tools and does not provide arguments to the model. See
[01-system-overview.md](01-system-overview.md) for the request flow.

## Public Artifacts

| Artifact | Location |
| --- | --- |
| 8.79B agent model | `spkc83/retail-bank-agent-9b` |
| Tool-use SFT dataset | `spkc83/retail-bank-agent-sft` |
| Dual-head router | `spkc83/retail-bank-domain-intent-router` |
| Router dataset | `spkc83/retail-bank-router-training-data` |
| Public POC Space | `spkc83/retail-bank-servicing-poc` |

The same artifact IDs appear in the root [README](../README.md), the model
cards under [../model_cards](../model_cards), and the data cards under
[../data_cards](../data_cards).

## Read In This Order

1. [System overview](01-system-overview.md)

   Learn the component boundaries, runtime request path, and where each part
   lives in the repository.

2. [Data generation](02-data-generation.md)

   Learn how the governed synthetic tool-use dataset is generated, how the
   classifier-only Banking77 and CLINC data is prepared, and which files prove
   provenance.

3. [Model and PEFT](03-model-and-peft.md)

   Learn the Granite base model identity, LoRA target modules, assistant-only
   masking, Granite tool wire, and merged/adapted release layout.

4. [Training, continuation, and recovery](04-training-and-recovery.md)

   Learn the guarded local and paid-job paths, checkpoint resume, continuation,
   export recovery, merge parity, and publication gates.

5. [Dual-head router](05-dual-head-router.md)

   Learn the shared DistilBERT encoder, domain and intent heads, governed data,
   calibration, release gates, and serving thresholds.

6. [Frozen evaluation](06-evaluation.md)

   Learn the two-phase frozen evaluation contract and the exact metrics needed
   for release.

7. [Inference and ZeroGPU POC](07-inference-and-poc.md)

   Learn model loading, token-budgeted history, static demo auth, the
   model-owned tool loop, synthetic SQLite state, and inference diagnostics.

8. [End-to-end runbook](08-end-to-end-runbook.md)

   Follow the complete install, data, training, evaluation, local POC, and
   deployment sequence.

Use the [file map](reference/file-map.md) to jump from concepts to code and the
[artifact ledger](reference/artifacts.md) for immutable revisions and hashes.

## Repository Map

| Path | Purpose |
| --- | --- |
| [../configs/banking-tool-sft-granite.toml](../configs/banking-tool-sft-granite.toml) | Granite PEFT training configuration. |
| [../data/banking-v3-tool-sft](../data/banking-v3-tool-sft) | Generated local copy of the tool-use SFT dataset. |
| [../data/banking-router-v1](../data/banking-router-v1) | Generated local copy of the router training data. |
| [../data/sources](../data/sources) | Tracked source locks for governed data preparation. |
| [../data_cards](../data_cards) | Public dataset documentation. |
| [../model_cards](../model_cards) | Public model documentation. |
| [../poc/retail-bank-customer-service-poc](../poc/retail-bank-customer-service-poc) | Gradio/ZeroGPU customer-service POC. |
| [../scripts/retail_bank](../scripts/retail_bank) | Data, training, recovery, evaluation, and Hub job entry points. |
| [../src/hello_slm](../src/hello_slm) | Shared package code for data, tool wire, evaluator, and router. |
| [../tests](../tests) | Repository regression tests. |

The active scripts use the non-versioned `scripts/retail_bank` path. The
retained `data/banking-v3-tool-sft` name is a published dataset-schema
identifier, not an alternate model implementation.

## Local Setup

Use Python 3.11 or newer for the repository package.

```bash
python -m pip install -e '.[dev]'
```

Install the larger training stack only when you need local or remote model
training helpers. The optional extra is named `scale` in
[../pyproject.toml](../pyproject.toml).

## Common Verification Commands

Run the targeted repository checks from the root directory.

```bash
python -m pytest -q \
  tests/test_banking_tool_wire.py \
  tests/test_banking_tool_sft_worker.py \
  tests/test_banking_router_data.py
```

Run the POC checks without loading the 8.79B model or router:

```bash
POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
  pytest -q poc/retail-bank-customer-service-poc/tests
```

Broader release checks are listed in the root [README](../README.md). Remote
Hugging Face Jobs and live ZeroGPU inference also require credentials, GPU
allocation, and exact revisions from the relevant script arguments.
