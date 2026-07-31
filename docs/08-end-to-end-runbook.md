# End-to-End Runbook

This runbook walks a new developer from local setup through data preparation,
router training, Granite tool SFT, frozen evaluation, and the ZeroGPU POC. It
uses the current active artifact set only.

The released reproduction remains below. For the unpublished v4 classifier
and Granite servicing-alignment candidates, follow
[Conversation Router v4](09-conversation-router-v4.md) and
[Granite Servicing Alignment v4](10-servicing-alignment-v4.md) instead of
overwriting released Hub repositories.

Paid Hugging Face Jobs steps are marked clearly. They are commands to run only
after explicit authorization, working credentials, and budget approval. This
document does not imply that paid jobs or public deployments have been started.

## 1. Install

From the repository root:

```bash
python -m pip install -e ".[dev,scale]"
```

The top-level package metadata is in [`pyproject.toml`](../pyproject.toml). The
POC has its own dependency set in
[`poc/retail-bank-customer-service-poc/pyproject.toml`](../poc/retail-bank-customer-service-poc/pyproject.toml)
and [`requirements.txt`](../poc/retail-bank-customer-service-poc/requirements.txt).

For POC-only work:

```bash
cd poc/retail-bank-customer-service-poc
python -m pip install -r requirements.txt
python -m pip install pytest ruff
```

## 2. Verify the Repo Locally

Run the focused repository tests:

```bash
python -m pytest -q tests
```

Run the POC tests without loading the 9B model or router:

```bash
cd poc/retail-bank-customer-service-poc
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-more"}'
export POC_SKIP_MODEL_LOAD=1
export POC_SKIP_ROUTER_LOAD=1
python -m pytest -q tests
```

Run static checks from the repository root:

```bash
ruff check .
MYPYPATH=src mypy src scripts tests
uv lock --check
```

If a command fails, fix that stage before moving downstream.

## 3. Prepare Router Data

The router data script is
[`scripts/retail_bank/prepare_dual_head_router_data.py`](../scripts/retail_bank/prepare_dual_head_router_data.py) for the
released v1 classifier, and [`scripts/retail_bank/prepare_conversation_router_data.py`](../scripts/retail_bank/prepare_conversation_router_data.py) for the
v4 candidate history-aware router.

Run:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_dual_head_router_data.py \
  --output-dir data/banking-router-v1
```

Optional v4 candidate data preparation:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_conversation_router_data.py \
  --output-dir data/banking-conversation-router-v4
```

Expected local outputs:

- [`data/banking-router-v1/manifest.json`](../data/banking-router-v1/manifest.json)
- [`data/banking-router-v1/train.jsonl`](../data/banking-router-v1/train.jsonl)
- [`data/banking-router-v1/validation.jsonl`](../data/banking-router-v1/validation.jsonl)
- [`data/banking-router-v1/test.jsonl`](../data/banking-router-v1/test.jsonl)
- [`data/banking-router-v1/README.md`](../data/banking-router-v1/README.md)

Current prepared split hashes:

| Split | Rows | SHA-256 |
| --- | ---: | --- |
| train | 44,432 | `c9067a04cefa90ed6fd874a6ebabbccb8015122b7920608c6d051df77b9f1acd` |
| validation | 8,589 | `7c1315a5f8555168143a0800364a98a6ce4f88ec7cb15b42216b15a83f33cfc5` |
| test | 16,260 | `e9178afc36ac2d90eef0587b4d42e2785e1c12e0a23445afe56486c3f61ac431` |

The published router dataset revision is
`54ff186a03501d76dc643dbed3d82729267ce811`.

## 4. Train and Publish the Router

The router trainer is
[`scripts/retail_bank/train_dual_head_router.py`](../scripts/retail_bank/train_dual_head_router.py).
It loads the published governed router dataset revision
`54ff186a03501d76dc643dbed3d82729267ce811`, trains a shared DistilBERT encoder
with domain and intent heads, calibrates the domain threshold, checks release
gates, writes a manifest, and publishes `spkc83/retail-bank-domain-intent-router`.

This step writes to Hugging Face Hub and requires `HF_TOKEN`:

```bash
export HF_TOKEN=...
uv run scripts/retail_bank/train_dual_head_router.py
```

The script has no dry-run CLI mode. Do not run it unless publishing the router
is intended.

For the v4 branch, use `scripts/retail_bank/train_conversation_router.py` with
`data/sources/banking-conversation-router-v4.lock.json` as the release lock and
`scripts/retail_bank/train_conversation_router.py`/`prepare_conversation_router_data.py` as the training pair.

Current released router:

- repo: `spkc83/retail-bank-domain-intent-router`
- revision: `136ee159d19cda7f585dd122907bbeb1ef4ec4db`
- calibrated lower boundary: `0.165`
- serving in-domain boundary: `0.50`
- model card: [`model_cards/retail-bank-domain-intent-router.md`](../model_cards/retail-bank-domain-intent-router.md)

The POC loads and verifies this artifact in
[`poc/retail-bank-customer-service-poc/router.py`](../poc/retail-bank-customer-service-poc/router.py).

## 5. Prepare Tool-Use SFT Data

The SFT data script is
[`scripts/retail_bank/prepare_tool_sft_data.py`](../scripts/retail_bank/prepare_tool_sft_data.py),
which delegates to
[`src/hello_slm/banking_tool_sft_data.py`](../src/hello_slm/banking_tool_sft_data.py).

Run:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_tool_sft_data.py \
  --output-dir data/banking-v3-tool-sft \
  --pilot-count 9000 \
  --split-seed 711
```

Current local outputs:

- [`data/banking-v3-tool-sft/manifest.json`](../data/banking-v3-tool-sft/manifest.json)
- [`data/banking-v3-tool-sft/train.jsonl`](../data/banking-v3-tool-sft/train.jsonl)
- [`data/banking-v3-tool-sft/validation.jsonl`](../data/banking-v3-tool-sft/validation.jsonl)
- [`data/banking-v3-tool-sft/test.jsonl`](../data/banking-v3-tool-sft/test.jsonl)
- [`data/banking-v3-tool-sft/README.md`](../data/banking-v3-tool-sft/README.md)

Current split identity:

| Split | Records | SHA-256 |
| --- | ---: | --- |
| train | 6,304 | `8d92fa0ab1d39875f0c4d918bc5aeaf670f71bf660a01a0a376cb4edc1cced53` |
| validation | 1,349 | `a8c7871b33689fce026ea570ad0a8a90a609cde232a89486e5437b028279e6d3` |
| test | 1,347 | `76b485fa507d56002f12b556f100fd842c77146804cf49be3426be031cc692c0` |

The manifest's public tool-manifest hash is
`sha256:88b6f53e19779732cde99190ebb5405d317e5691f91bc63624ef32c241939b40`.
The published SFT dataset revision is
`183e7e1ed1aba9c3d7155e7b83b64dc854935055`.

## 6. Inspect the Granite Training Plan

The active training config is
[`configs/banking-tool-sft-granite.toml`](../configs/banking-tool-sft-granite.toml).
The guarded worker is
[`scripts/retail_bank/cloud_train_tool_sft.py`](../scripts/retail_bank/cloud_train_tool_sft.py).

Dry-run inspection does not start a paid job, download 9B weights, merge a
checkpoint, or push to the Hub:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
  --manifest data/banking-v3-tool-sft/manifest.json
```

The plan should show:

- base model: `ibm-granite/granite-4.1-8b`
- base revision: `1504002f650e656a0a3789d99574df12e3e94ed0`
- family: `granite`
- destination: `spkc83/retail-bank-agent-9b`
- LoRA rank 32, alpha 64, dropout 0.05
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`
- max sequence length: 2,048
- remote guard requiring `--execute-remote`, `--allow-remote-execution`, and
  `RETAIL_BANK_ALLOW_REMOTE_TOOL_SFT=banking-v3-tool-sft`

For a local offline smoke path, use the worker's `--run-tiny-smoke` option. That
path uses a small stand-in tokenizer/model path and does not launch a cloud job.

### 6b. V4 Servicing-Alignment Continuation Candidate (9B start)

The v4 candidate run starts from the released merged checkpoint and appends
conversation-aligned records from `data/banking-servicing-alignment-v4`:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
  --manifest data/banking-servicing-alignment-v4/manifest.json \
  --base-model spkc83/retail-bank-agent-9b \
  --base-revision 085df3d089cfadd77424b548542da0390a54a23e \
  --family granite \
  --hub-dest spkc83/retail-bank-servicing-agent-9b \
  --learning-rate 2e-5 \
  --max-steps 500 \
  --max-train-seconds 14400 \
  --dry-run
```

To launch this in HF Jobs on the same 5-hour RTX PRO 6000 path, set:

```bash
export BASE_MODEL=spkc83/retail-bank-agent-9b
export BASE_REVISION=085df3d089cfadd77424b548542da0390a54a23e
export CONFIRMATION_TOKEN=banking-v3-tool-sft
export DATASET_REPO=spkc83/retail-bank-servicing-alignment-sft
export HF_HUB_DEST=spkc83/retail-bank-servicing-agent-9b
export MAX_STEPS=500
export LEARNING_RATE=2e-5
export CHECKPOINT_EVERY=100
export TRACKIO_PROJECT=retail-bank-servicing-v4
export PROJECT_LABEL=retail-bank-servicing-v4
export OUTPUT_PREFIX=/data/retail-bank-servicing-agent-9b-v4
scripts/retail_bank/run_remote_training_job.sh \
  "$(git rev-parse HEAD)" \
  DATASET_REVISION_40_HEX
```

This invokes `cloud_train_tool_sft.py` with a fresh LoRA over the released
merged weights. It does not resume the retained v3 adapter or optimizer state.
Then run the candidate through the frozen evaluator and promote only when every
exact tool, response-path, grounding, and zero-error gate passes.

## 7. Paid Granite Training Job

This is a paid Hugging Face Jobs step. Run it only after explicit authorization,
valid `HF_TOKEN`, and budget approval.

Launcher:
[`scripts/retail_bank/run_remote_training_job.sh`](../scripts/retail_bank/run_remote_training_job.sh)

Command shape:

```bash
scripts/retail_bank/run_remote_training_job.sh \
  "$(git rev-parse HEAD)" \
  183e7e1ed1aba9c3d7155e7b83b64dc854935055
```

Commit and push the exact source state first. The launcher validates the
result of `git rev-parse HEAD`, checks that the job script exists at that exact
GitHub commit, and then calls `hf jobs uv run` with:

- flavor: `rtx-pro-6000`
- timeout: `5h`
- secret: `HF_TOKEN`
- volume: `hf://buckets/spkc83/jobs-artifacts:/data`
- job script: [`hf_job_tool_sft.py`](../scripts/retail_bank/hf_job_tool_sft.py)

Inside the job, the worker trains BF16 LoRA on Granite, checkpoints, optionally
merges the adapter, verifies reload behavior, and pushes only because the job
passes the explicit remote guard and `--push-to-hub`.

Current released model identity:

- repo: `spkc83/retail-bank-agent-9b`
- immutable weights revision: `085df3d089cfadd77424b548542da0390a54a23e`
- training/provenance revision: `247ac402989144698f89727a59a07ce5d05f31c6`
- published evaluation head: `98cde9ee058b785fb871abcd2c85e18cea410bdf`
- training job: `spkc83/6a6a60d4b36a6516e96a0709`
- FP16-native recovery and merge-parity job:
  `spkc83/6a6a6b6323ed89c748ec502c`
- model card: [`model_cards/retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md)

## 8. Paid Frozen Tool Evaluation

This is also a paid Hugging Face Jobs step. Run it only after explicit
authorization, valid `HF_TOKEN`, and budget approval.

Launcher:
[`scripts/retail_bank/run_remote_tool_eval_job.sh`](../scripts/retail_bank/run_remote_tool_eval_job.sh)

Command shape for the released model:

```bash
bash scripts/retail_bank/run_remote_tool_eval_job.sh \
  "$(git rev-parse HEAD)" \
  085df3d089cfadd77424b548542da0390a54a23e \
  183e7e1ed1aba9c3d7155e7b83b64dc854935055
```

The launcher validates all three revisions as exact 40-character lowercase Git
commits, checks that the remote job script exists, and then calls `hf jobs uv
run` with:

- flavor: `rtx-pro-6000`
- timeout: `2h`
- secret: `HF_TOKEN`
- volume: `hf://buckets/spkc83/jobs-artifacts:/data`
- job script: [`hf_job_tool_eval.py`](../scripts/retail_bank/hf_job_tool_eval.py)

The eval worker
[`cloud_generate_tool_eval.py`](../scripts/retail_bank/cloud_generate_tool_eval.py)
generates deterministic predictions against the frozen test split and scores
them with [`src/hello_slm/banking_tool_eval.py`](../src/hello_slm/banking_tool_eval.py).

Released evaluation job:

- job: `spkc83/6a6a6c7cb36a6516e96a0ac4`
- report path in model repo:
  `evaluation/085df3d089cf-183e7e1ed1ab/`
- frozen records: 1,347
- tool names and arguments: `774/774`
- executable tool trajectories: `678/678`
- dependent multi-tool sequences: `96/96`
- clarifications: `63/63`
- banking FAQs: `258/258`
- OOD paths: `30/30`
- grounded factual responses: `1,119/1,119`

The released evaluation files are retained in the model repository. The
private bucket was only a restartable staging location; its released
evaluation copy and obsolete training intermediates were retired on
2026-07-31. The selected continuation step-600 recovery checkpoint remains.
See [`docs/04-training-and-recovery.md`](04-training-and-recovery.md) for the
exact retention policy.

## 9. Run the POC Locally

Use local skip flags for UI, auth, routing stubs, state, and non-model plumbing:

```bash
cd poc/retail-bank-customer-service-poc
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-more"}'
export POC_SKIP_MODEL_LOAD=1
export POC_SKIP_ROUTER_LOAD=1
python app.py
```

This starts the Gradio app but cannot prove 9B inference because the model and
router are intentionally skipped.

For a live model path, run in an environment with CUDA and the required memory,
remove `POC_SKIP_MODEL_LOAD`, and keep:

```bash
export RETAIL_BANK_MODEL_ID=spkc83/retail-bank-agent-9b
export RETAIL_BANK_MODEL_REVISION=085df3d089cfadd77424b548542da0390a54a23e
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-more"}'
```

The public Space runs this path through
[`zero_gpu_runtime.py`](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py)
and the `@spaces.GPU(size="large", duration=90)` boundary in
[`app.py`](../poc/retail-bank-customer-service-poc/app.py).

## 10. Deploy the Space

Deployment to `spkc83/retail-bank-servicing-poc` is an external production
action. Do not deploy without explicit authorization.

Files to publish are under
[`poc/retail-bank-customer-service-poc/`](../poc/retail-bank-customer-service-poc/).
The Space card is
[`poc/retail-bank-customer-service-poc/README.md`](../poc/retail-bank-customer-service-poc/README.md).

Deployment is performed from the linked GitHub Space source repository (`github-poc` remote in this project). Keep it aligned with this branch and this exact checkout before publishing a deployment change.

Before deployment:

1. Run the POC tests with skip flags.
2. Verify [`requirements.txt`](../poc/retail-bank-customer-service-poc/requirements.txt).
3. Confirm the model revision in
   [`zero_gpu_runtime.py`](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py).
4. Confirm the router revision in
   [`router.py`](../poc/retail-bank-customer-service-poc/router.py).
5. Set the Space secret `DEMO_AUTH_JSON` to exactly the two demo usernames.

After deployment, use the diagnostics panel to verify:

- model `spkc83/retail-bank-agent-9b`
- exact revision `085df3d089cfadd77424b548542da0390a54a23e`
- registered execution boundary `ZeroGPU large`
- CUDA runtime device for model-handled turns
- no model passes for high-confidence OOD turns

Run at least these live scenarios:

- greeting or small talk
- account balance read
- transaction list read
- card freeze write
- card replacement write
- transaction dispute write
- pending transfer cancellation
- dependent multi-tool follow-up
- ambiguous clarification
- banking FAQ
- out-of-domain prompt
- multi-turn follow-up using previous context

A live deployment should be treated as successful when both of these hold:

- every local POC preflight test passes in skip mode, and diagnostics proves that model-handled turns use `spkc83/retail-bank-agent-9b` at the pinned revision;
- OOD responses are stable for unsupported banking prompts and do not trigger tool-call syntax errors.

Stop when all local tests pass and live diagnostics prove the active revision
for model-handled turns.
