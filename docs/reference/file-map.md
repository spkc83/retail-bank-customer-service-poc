# File Map

This map links the current active retail-bank agent workflow to repo files.

## Root

| Path | Purpose |
| --- | --- |
| [`../../README.md`](../../README.md) | Project overview, active public artifacts, runtime summary, verification commands. |
| [`../../pyproject.toml`](../../pyproject.toml) | Top-level package metadata, dev and scale dependencies, pytest, mypy, and ruff settings. |
| [`../../uv.lock`](../../uv.lock) | Locked top-level Python environment. |
| [`../../Makefile`](../../Makefile) | Repo command shortcuts, when used. |

## Active Docs

| Path | Purpose |
| --- | --- |
| [`../07-inference-and-poc.md`](../07-inference-and-poc.md) | Detailed POC inference, routing, auth, tool loop, diagnostics, and deployment guide. |
| [`../08-end-to-end-runbook.md`](../08-end-to-end-runbook.md) | Install-to-data-to-training-to-eval-to-POC runbook. |
| [`../09-conversation-router-v4.md`](../09-conversation-router-v4.md) | History-aware cross-encoder candidate, leakage-safe data, local training, release gates, and POC integration. |
| [`../10-servicing-alignment-v4.md`](../10-servicing-alignment-v4.md) | Composite Granite continuation-SFT design, use-case coverage, safe training plan, and release stop condition. |
| [`artifacts.md`](artifacts.md) | Immutable model, dataset, job, and split identity ledger. |

## Published Cards

| Path | Purpose |
| --- | --- |
| [`../../data_cards/retail-bank-agent-sft.md`](../../data_cards/retail-bank-agent-sft.md) | Published tool-use SFT dataset card. |
| [`../../data_cards/retail-bank-router-training-data.md`](../../data_cards/retail-bank-router-training-data.md) | Published router-training dataset card. |
| [`../../data_cards/retail-bank-servicing-alignment-sft.md`](../../data_cards/retail-bank-servicing-alignment-sft.md) | Candidate composite Granite servicing-alignment dataset card. |
| [`../../model_cards/retail-bank-agent-9b.md`](../../model_cards/retail-bank-agent-9b.md) | Published Granite agent model card and released evaluation results. |
| [`../../model_cards/retail-bank-domain-intent-router.md`](../../model_cards/retail-bank-domain-intent-router.md) | Published dual-head router card and serving thresholds. |

## Configuration

| Path | Purpose |
| --- | --- |
| [`../../configs/banking-tool-sft-granite.toml`](../../configs/banking-tool-sft-granite.toml) | Active Granite BF16 LoRA training configuration. |

## Local Data

| Path | Purpose |
| --- | --- |
| [`../../data/banking-v3-tool-sft/manifest.json`](../../data/banking-v3-tool-sft/manifest.json) | Local tool-use SFT split manifest. |
| [`../../data/banking-v3-tool-sft/train.jsonl`](../../data/banking-v3-tool-sft/train.jsonl) | Local SFT training split. |
| [`../../data/banking-v3-tool-sft/validation.jsonl`](../../data/banking-v3-tool-sft/validation.jsonl) | Local SFT validation split. |
| [`../../data/banking-v3-tool-sft/test.jsonl`](../../data/banking-v3-tool-sft/test.jsonl) | Local frozen SFT test split. |
| [`../../data/banking-v3-tool-sft/README.md`](../../data/banking-v3-tool-sft/README.md) | Local SFT dataset README/card. |
| [`../../data/banking-router-v1/manifest.json`](../../data/banking-router-v1/manifest.json) | Local router dataset manifest. |
| [`../../data/banking-router-v1/train.jsonl`](../../data/banking-router-v1/train.jsonl) | Local router training split. |
| [`../../data/banking-router-v1/validation.jsonl`](../../data/banking-router-v1/validation.jsonl) | Local router validation/calibration split. |
| [`../../data/banking-router-v1/test.jsonl`](../../data/banking-router-v1/test.jsonl) | Local router test split. |
| [`../../data/sources/banking-router-v1.lock.json`](../../data/sources/banking-router-v1.lock.json) | Router source and prepared split digest lock. |
| [`../../data/sources/banking-conversation-router-v4.lock.json`](../../data/sources/banking-conversation-router-v4.lock.json) | Candidate cross-encoder source and deterministic prepared-split lock. |
| [`../../data/sources/banking-servicing-alignment-v4.lock.json`](../../data/sources/banking-servicing-alignment-v4.lock.json) | Candidate composite Granite data base-manifest and split-digest lock. |

## Source Package

| Path | Purpose |
| --- | --- |
| [`../../src/hello_slm/banking_tool_sft_data.py`](../../src/hello_slm/banking_tool_sft_data.py) | Tool-use SFT data generator, public tool manifest, validators. |
| [`../../src/hello_slm/banking_tool_wire.py`](../../src/hello_slm/banking_tool_wire.py) | Tool-wire adapter used by training and evaluation. |
| [`../../src/hello_slm/banking_tool_eval.py`](../../src/hello_slm/banking_tool_eval.py) | Frozen tool/final-response evaluator. |
| [`../../src/hello_slm/banking_router_data.py`](../../src/hello_slm/banking_router_data.py) | Router split builder and data policy helpers. |
| [`../../src/hello_slm/banking_dual_head_router.py`](../../src/hello_slm/banking_dual_head_router.py) | Shared dual-head router code used by tests and scripts. |
| [`../../src/hello_slm/banking_conversation_router.py`](../../src/hello_slm/banking_conversation_router.py) | Candidate shared-encoder, three-head classifier model. |
| [`../../src/hello_slm/banking_conversation_router_data.py`](../../src/hello_slm/banking_conversation_router_data.py) | Leakage-safe history rendering and deterministic v4 classifier split builder. |
| [`../../src/hello_slm/banking_servicing_alignment_data.py`](../../src/hello_slm/banking_servicing_alignment_data.py) | Composite Granite SFT alignment generator and validation policy. |
| [`../../src/hello_slm/config.py`](../../src/hello_slm/config.py) | Canonical JSON and config helpers. |

## Scripts

| Path | Purpose |
| --- | --- |
| [`../../scripts/retail_bank/prepare_tool_sft_data.py`](../../scripts/retail_bank/prepare_tool_sft_data.py) | CLI wrapper for tool-use SFT data preparation. |
| [`../../scripts/retail_bank/prepare_dual_head_router_data.py`](../../scripts/retail_bank/prepare_dual_head_router_data.py) | CLI for governed Banking77 and CLINC150 router data preparation. |
| [`../../scripts/retail_bank/train_dual_head_router.py`](../../scripts/retail_bank/train_dual_head_router.py) | Router train/calibrate/evaluate/publish script. |
| [`../../scripts/retail_bank/prepare_conversation_router_data.py`](../../scripts/retail_bank/prepare_conversation_router_data.py) | Candidate history-aware router data preparation and digest verification. |
| [`../../scripts/retail_bank/train_conversation_router.py`](../../scripts/retail_bank/train_conversation_router.py) | Local candidate cross-encoder training, calibration, test gates, artifact writing, and optional publication. |
| [`../../scripts/retail_bank/prepare_servicing_alignment_data.py`](../../scripts/retail_bank/prepare_servicing_alignment_data.py) | Candidate composite Granite data preparation, lock verification, and optional explicit publication. |
| [`../../scripts/retail_bank/cloud_train_tool_sft.py`](../../scripts/retail_bank/cloud_train_tool_sft.py) | Guarded local/remote Granite tool-SFT worker. |
| [`../../scripts/retail_bank/hf_job_tool_sft.py`](../../scripts/retail_bank/hf_job_tool_sft.py) | Hugging Face Jobs bootstrap for paid Granite SFT. |
| [`../../scripts/retail_bank/run_remote_training_job.sh`](../../scripts/retail_bank/run_remote_training_job.sh) | Paid Granite training job launcher. |
| [`../../scripts/retail_bank/cloud_generate_tool_eval.py`](../../scripts/retail_bank/cloud_generate_tool_eval.py) | Frozen prediction generation and scoring worker. |
| [`../../scripts/retail_bank/hf_job_tool_eval.py`](../../scripts/retail_bank/hf_job_tool_eval.py) | Hugging Face Jobs bootstrap for paid frozen eval. |
| [`../../scripts/retail_bank/run_remote_tool_eval_job.sh`](../../scripts/retail_bank/run_remote_tool_eval_job.sh) | Paid frozen eval job launcher. |
| [`../../scripts/retail_bank/hf_job_remerge_tool_sft.py`](../../scripts/retail_bank/hf_job_remerge_tool_sft.py) | Merge recovery helper used by release validation. |
| [`../../scripts/retail_bank/hf_job_merge_parity.py`](../../scripts/retail_bank/hf_job_merge_parity.py) | Merge parity helper. |

## POC

| Path | Purpose |
| --- | --- |
| [`../../poc/retail-bank-customer-service-poc/README.md`](../../poc/retail-bank-customer-service-poc/README.md) | Hugging Face Space card and public POC docs. |
| [`../../poc/retail-bank-customer-service-poc/app.py`](../../poc/retail-bank-customer-service-poc/app.py) | Gradio app, routing, ZeroGPU event, diagnostics, UI. |
| [`../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) | Granite model/tokenizer loading, token counting, deterministic generation. |
| [`../../poc/retail-bank-customer-service-poc/model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) | Model-owned tool loop, prompt budgeting, tool parsing, validation, execution trace. |
| [`../../poc/retail-bank-customer-service-poc/router.py`](../../poc/retail-bank-customer-service-poc/router.py) | Candidate CPU cross-encoder loading, artifact verification, history rendering, three heads, and calibrated routing. |
| [`../../poc/retail-bank-customer-service-poc/auth.py`](../../poc/retail-bank-customer-service-poc/auth.py) | Static demo auth loader. |
| [`../../poc/retail-bank-customer-service-poc/mock_bank.py`](../../poc/retail-bank-customer-service-poc/mock_bank.py) | Session-isolated SQLite synthetic bank backend and tool implementation. |
| [`../../poc/retail-bank-customer-service-poc/state.py`](../../poc/retail-bank-customer-service-poc/state.py) | Session registry initialization. |
| [`../../poc/retail-bank-customer-service-poc/responses.py`](../../poc/retail-bank-customer-service-poc/responses.py) | Stock OOD and model-failure responses. |
| [`../../poc/retail-bank-customer-service-poc/synthetic_bank.json`](../../poc/retail-bank-customer-service-poc/synthetic_bank.json) | Synthetic customer seed records. |
| [`../../poc/retail-bank-customer-service-poc/requirements.txt`](../../poc/retail-bank-customer-service-poc/requirements.txt) | Space dependency list. |
| [`../../poc/retail-bank-customer-service-poc/pyproject.toml`](../../poc/retail-bank-customer-service-poc/pyproject.toml) | POC package metadata and local test settings. |

## Tests

| Path | Purpose |
| --- | --- |
| [`../../tests/test_banking_tool_sft_data.py`](../../tests/test_banking_tool_sft_data.py) | SFT data-generation tests. |
| [`../../tests/test_banking_tool_wire.py`](../../tests/test_banking_tool_wire.py) | Tool-wire adapter tests. |
| [`../../tests/test_banking_tool_eval.py`](../../tests/test_banking_tool_eval.py) | Evaluator tests. |
| [`../../tests/test_banking_tool_eval_runner.py`](../../tests/test_banking_tool_eval_runner.py) | Frozen eval runner and launcher tests. |
| [`../../tests/test_banking_router_preparation.py`](../../tests/test_banking_router_preparation.py) | Router data preparation tests. |
| [`../../tests/test_banking_router_training.py`](../../tests/test_banking_router_training.py) | Router training and release-gate tests. |
| [`../../tests/test_banking_dual_head_router.py`](../../tests/test_banking_dual_head_router.py) | Shared router behavior tests. |
| [`../../tests/test_banking_conversation_router.py`](../../tests/test_banking_conversation_router.py) | Candidate three-head model tests. |
| [`../../tests/test_banking_conversation_router_data.py`](../../tests/test_banking_conversation_router_data.py) | Candidate history rendering, split isolation, leakage, and held-out data tests. |
| [`../../tests/test_banking_conversation_router_preparation.py`](../../tests/test_banking_conversation_router_preparation.py) | Candidate source-lock and deterministic preparation tests. |
| [`../../tests/test_banking_conversation_router_training.py`](../../tests/test_banking_conversation_router_training.py) | Candidate routing policy, metric, calibration, and release-gate tests. |
| [`../../tests/test_banking_servicing_alignment_data.py`](../../tests/test_banking_servicing_alignment_data.py) | Candidate Granite alignment coverage, composite counts, held-out isolation, and lock-drift tests. |
| [`../../tests/test_banking_tool_sft_worker.py`](../../tests/test_banking_tool_sft_worker.py) | Tool SFT worker tests. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_auth.py`](../../poc/retail-bank-customer-service-poc/tests/test_auth.py) | POC auth tests. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_router.py`](../../poc/retail-bank-customer-service-poc/tests/test_router.py) | POC router tests. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_model_service.py`](../../poc/retail-bank-customer-service-poc/tests/test_model_service.py) | POC model-service and tool-loop tests. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_mock_bank.py`](../../poc/retail-bank-customer-service-poc/tests/test_mock_bank.py) | SQLite synthetic backend tests. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_app.py`](../../poc/retail-bank-customer-service-poc/tests/test_app.py) | Gradio app behavior tests. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/tests/test_zero_gpu_runtime.py) | ZeroGPU skip/runtime helper tests. |
