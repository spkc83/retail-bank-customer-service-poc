# Granite PEFT Training and Recovery

This runbook covers the active IBM Granite 8.79B PEFT lane only: local checks, initial training, continuation, export recovery, remerge, merge parity, and final publication. The training configuration is [`configs/banking-tool-sft-granite.toml`](../configs/banking-tool-sft-granite.toml). The published model card is [`model_cards/retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md).

## Active Artifact IDs

Use immutable 40-character revisions for every paid or published run. Branch names such as `main` are rejected by the job entry points.

| Artifact | Value | Owner |
| --- | --- | --- |
| Base model | `ibm-granite/granite-4.1-8b` | [`configs/banking-tool-sft-granite.toml`](../configs/banking-tool-sft-granite.toml) |
| Base revision | `1504002f650e656a0a3789d99574df12e3e94ed0` | [`configs/banking-tool-sft-granite.toml`](../configs/banking-tool-sft-granite.toml) |
| Model repo | `spkc83/retail-bank-agent-9b` | [`scripts/retail_bank/hf_job_tool_sft.py`](../scripts/retail_bank/hf_job_tool_sft.py) |
| Released weights revision | `085df3d089cfadd77424b548542da0390a54a23e` | [`model_cards/retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md) |
| Training dataset repo | `spkc83/retail-bank-agent-sft` | [`data_cards/retail-bank-agent-sft.md`](../data_cards/retail-bank-agent-sft.md) |
| Training dataset revision | `183e7e1ed1aba9c3d7155e7b83b64dc854935055` | [`data_cards/retail-bank-agent-sft.md`](../data_cards/retail-bank-agent-sft.md) |
| Source revision used for release | `4270636255515f7a563d935794a3642e0b13ccb3` | [`model_cards/retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md) |
| Recovery source revision | `0237b97c0a9558bbb2e95c45097ac5ae5f9f7f21` | [`model_cards/retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md) |
| Training job | `spkc83/6a6a60d4b36a6516e96a0709` | [`model_cards/retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md) |
| Recovery and parity job | `spkc83/6a6a6b6323ed89c748ec502c` | [`model_cards/retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md) |

## Bucket Retention Policy

The private `spkc83/jobs-artifacts` bucket is durable working storage for an
active job, but it is not the release source of truth. On 2026-07-31, after the
merged weights, adapter, provenance, and frozen evaluation were verified in
their published Hub repositories, 232 obsolete job files totaling 448.2 GB
were removed. The bucket now retains 58 files totaling about 1.25 GB.

The retained recovery set is:

- `retail-bank-agent-9b-continuation-42706362-68a4e5b1/checkpoint-600/`;
- `continuation_training_metadata.json` and the small JSON provenance records
  for that released continuation.

This preserves export recovery for the selected step-600 adapter. Superseded
checkpoints, failed-run outputs, duplicate merged 16.8 GB weights, temporary
merge files, optimizer state from non-selected runs, and bucket copies of
published evaluation results were retired. The active model, adapter,
evaluation, router, datasets, and Space do not load from this bucket.

New training and evaluation runs must use a new output prefix. Their
intermediate files may be removed after publication and verification, but the
selected recovery adapter, trainer state, and run metadata must remain until
that release is formally retired.

## What Trains

The worker [`scripts/retail_bank/cloud_train_tool_sft.py`](../scripts/retail_bank/cloud_train_tool_sft.py) loads the pinned Granite base and trains a BF16 LoRA adapter with TRL `SFTTrainer`. The retained hyperparameters are:

| Setting | Value |
| --- | --- |
| PEFT stack | BF16 LoRA over Granite attention and MLP projections |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| LoRA rank / alpha / dropout | `32` / `64` / `0.05` |
| Maximum sequence length | `2048` |
| Initial full-job optimizer cap | `14400` seconds |
| Outer HF Jobs timeout | `5h` |
| Initial checkpoints | every `500` steps |
| Output root in paid job | `/data/retail-bank-agent-9b-${SOURCE_COMMIT:0:8}` |

The SFT corpus is declared in [`data/banking-v3-tool-sft/manifest.json`](../data/banking-v3-tool-sft/manifest.json) and summarized in [`data/banking-v3-tool-sft/preparation-report.json`](../data/banking-v3-tool-sft/preparation-report.json). It contains 6,304 training, 1,349 validation, and 1,347 frozen test conversations. The tokenizer and loss masking path is covered by [`src/hello_slm/banking_tool_wire.py`](../src/hello_slm/banking_tool_wire.py) and tests such as [`tests/test_banking_tool_wire.py`](../tests/test_banking_tool_wire.py).

## Local Preflight

Run these before any paid job:

```bash
python -m pytest -q tests/test_banking_tool_sft_data.py \
  tests/test_banking_tool_wire.py \
  tests/test_banking_tool_sft_job.py \
  tests/test_banking_tool_sft_worker.py \
  tests/test_banking_tool_sft_continuation.py \
  tests/test_banking_tool_sft_export_recovery.py \
  tests/test_banking_tool_sft_release.py
```

Check the training plan without downloading 9B weights, launching a job, merging, or pushing:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
  --manifest data/banking-v3-tool-sft/manifest.json \
  --base-revision 1504002f650e656a0a3789d99574df12e3e94ed0 \
  --family granite \
  --max-steps 3000 \
  --max-train-seconds 14400 \
  --checkpoint-every 500 \
  --dry-run
```

Run the local one-step smoke:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
  --run-tiny-smoke \
  --dry-run
```

The tiny smoke uses small offline stand-ins. It writes local smoke artifacts and verifies assistant-label tokens, checkpoint metadata, adapter output paths, final output paths, and merge/reload parity. It does not prove the 8.79B base can train on GPU.

## Dry Run vs Paid HF Jobs

Dry run is the default for [`cloud_train_tool_sft.py`](../scripts/retail_bank/cloud_train_tool_sft.py). It reports the intended stack and guard state. It refuses to:

- download the 9B Granite base weights;
- start paid/cloud work;
- write to Hugging Face Hub;
- merge or publish a checkpoint.

Paid execution is launched through [`scripts/retail_bank/run_remote_training_job.sh`](../scripts/retail_bank/run_remote_training_job.sh), which submits [`scripts/retail_bank/hf_job_tool_sft.py`](../scripts/retail_bank/hf_job_tool_sft.py) with:

- `--flavor rtx-pro-6000`;
- `--timeout 5h`;
- `--secrets HF_TOKEN`;
- `--volume hf://buckets/spkc83/jobs-artifacts:/data`;
- exact source and dataset revisions.

The `HF_TOKEN` secret must have read access to the dataset and base, and write access to `spkc83/retail-bank-agent-9b`. Do not put tokens on the command line.

```bash
scripts/retail_bank/run_remote_training_job.sh \
  "$(git rev-parse HEAD)" \
  183e7e1ed1aba9c3d7155e7b83b64dc854935055
```

The mounted bucket persists under `/data`, so checkpoints survive job exit
during an active run and can be reused by continuation or recovery until the
retention policy is applied. The wrapper builds the output path as
`/data/retail-bank-agent-9b-${SOURCE_COMMIT:0:8}`.

The active launchers and source of truth are under `scripts/retail_bank`.
For compatibility only, when resuming a source commit created before the
directory rename, a launcher probes the pre-rename bootstrap URL after the
current path is absent. The old directory is not part of the current tree.

## Resume Initial Training

If an active job exits after writing a retained checkpoint, resume with the
same source and dataset revisions plus the checkpoint path under the persisted
bucket:

```bash
scripts/retail_bank/run_remote_training_job.sh \
  SOURCE_COMMIT_40_HEX \
  183e7e1ed1aba9c3d7155e7b83b64dc854935055 \
  /data/retail-bank-agent-9b-SOURCE8/checkpoint-STEP
```

The resume flag is forwarded by [`hf_job_tool_sft.py`](../scripts/retail_bank/hf_job_tool_sft.py) to `cloud_train_tool_sft.py --resume-from`. Keep the source revision unchanged unless the purpose is an explicitly new run.

## Continuation Training

Continuation does not rerun the original SFT. [`scripts/retail_bank/cloud_continue_tool_sft.py`](../scripts/retail_bank/cloud_continue_tool_sft.py) loads the pinned Granite base plus the retained adapter from a pinned model revision, then oversamples sequential, clarification, and servicing-quality records while retaining single-tool, tool-error, FAQ, OOD, and hard-negative regression records. Its behavior is tested in [`tests/test_banking_tool_sft_continuation.py`](../tests/test_banking_tool_sft_continuation.py).

Dry-run the continuation plan:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_continue_tool_sft.py \
  --manifest data/banking-v3-tool-sft/manifest.json \
  --source-model-revision 085df3d089cfadd77424b548542da0390a54a23e \
  --base-revision 1504002f650e656a0a3789d99574df12e3e94ed0 \
  --max-steps 600 \
  --max-train-seconds 9000 \
  --checkpoint-every 100 \
  --dry-run
```

Launch the paid continuation:

```bash
scripts/retail_bank/run_remote_continuation_job.sh \
  SOURCE_COMMIT_40_HEX \
  183e7e1ed1aba9c3d7155e7b83b64dc854935055 \
  085df3d089cfadd77424b548542da0390a54a23e \
  600
```

The wrapper mounts the same durable bucket and writes `/data/retail-bank-agent-9b-continuation-${SOURCE_COMMIT:0:8}-${SOURCE_MODEL_REVISION:0:8}`. The job secret remains `HF_TOKEN`. The remote worker requires `RETAIL_BANK_ALLOW_REMOTE_CONTINUATION_SFT=banking-v3-continuation-sft`, which [`hf_job_continue_tool_sft.py`](../scripts/retail_bank/hf_job_continue_tool_sft.py) sets inside the job.

## Remerge and Merge Parity

The release path keeps the adapter and the merged root checkpoint separate:

- [`scripts/retail_bank/hf_job_remerge_tool_sft.py`](../scripts/retail_bank/hf_job_remerge_tool_sft.py) rebuilds a merged FP16 checkpoint from the adapter and pinned Granite base.
- [`scripts/retail_bank/hf_job_merge_parity.py`](../scripts/retail_bank/hf_job_merge_parity.py) compares adapter-vs-merged logits and deterministic generations on eight prompts.
- [`scripts/retail_bank/hf_job_finalize_tool_sft.py`](../scripts/retail_bank/hf_job_finalize_tool_sft.py) validates the parity report and publishes allowlisted files.

The finalizer gate, covered by [`tests/test_banking_tool_sft_release.py`](../tests/test_banking_tool_sft_release.py), requires:

| Gate | Threshold |
| --- | --- |
| Finite logit differences | all true |
| Greedy generations equal | all true |
| Compared prompts | at least `8` |
| Argmax token agreement | at least `0.999` |
| Maximum absolute logit drift | at most `0.3` |
| p999 absolute logit drift | at most `0.07` |

The release model card records the accepted FP16-native continuation values: `8/8` exact representative generations, argmax agreement `1.0`, mean absolute logit drift `0.00828770`, p99/p999 drift `0.0410156`/`0.0703125`, and maximum drift `0.238281`.

## Export Recovery

Use export recovery when continuation training completed and wrote a retained
adapter to the bucket, but publication or final export failed. The released
v3 step-600 adapter remains available for this path; superseded v3 checkpoints
do not. Recovery is export-only: it does not call `trainer.train`. The launcher
is [`scripts/retail_bank/run_remote_continuation_export_recovery.sh`](../scripts/retail_bank/run_remote_continuation_export_recovery.sh), the job bootstrap is [`scripts/retail_bank/hf_job_recover_continuation_export.py`](../scripts/retail_bank/hf_job_recover_continuation_export.py), and the recovery worker is [`scripts/retail_bank/cloud_recover_continuation_export.py`](../scripts/retail_bank/cloud_recover_continuation_export.py).

```bash
scripts/retail_bank/run_remote_continuation_export_recovery.sh \
  RECOVERY_SOURCE_COMMIT_40_HEX \
  TRAINING_SOURCE_COMMIT_40_HEX \
  183e7e1ed1aba9c3d7155e7b83b64dc854935055 \
  PARENT_MODEL_REVISION_40_HEX \
  TRAINING_JOB_ID \
  /data/retail-bank-agent-9b-continuation-SOURCE8-PARENT8 \
  SELECTED_STEP
```

Optional argument 8 overrides the selected adapter subdirectory; otherwise the wrapper uses `checkpoint-${SELECTED_STEP}`. The wrapper requires `OUTPUT_ROOT` to start with `/data/retail-bank-agent-9b-continuation-` and caps the export recovery job at `1h`.

Recovery cross-checks persisted metadata against the inspected training job:

- parent model revision;
- dataset revision;
- base model and base revision;
- training source commit;
- output root;
- completed step count;
- training-job artifact time window.

It tries the FP16-native candidate before the FP32-accumulated FP16 candidate and publishes only after unchanged parity gates pass. This is covered by [`tests/test_banking_tool_sft_export_recovery.py`](../tests/test_banking_tool_sft_export_recovery.py).

## Stop Conditions

Stop before paid training if any local preflight fails, if the dry-run guard says remote execution is not intentionally enabled, or if the source/dataset/base revision is not exact. Stop during or after remote work if:

- checkpoint metadata is missing or mismatched;
- validation loss or token accuracy is unavailable;
- merge/reload parity fails;
- merge parity gates fail;
- the job cannot write to `/data`;
- Hub upload cannot record exact published revisions;
- frozen evaluation in [`docs/06-evaluation.md`](06-evaluation.md) fails a release gate.

Do not use output repair, deterministic tool planning, or branch-name revisions to make a checkpoint appear releasable.

## Expected Outputs

Initial and continuation training write under the mounted `/data` output root:

- adapter files under `adapter/`;
- merged root checkpoint under the selected merged subdirectory;
- checkpoint metadata under `checkpoint-*` or `checkpoints/step-*`;
- `training_result.json`;
- remerge report such as `fp16_remerge.json`;
- merge parity report such as `merge_parity_diagnostics_merged-fp16_float16.json`;
- Hub evidence files including `merge_parity_diagnostics.json` and exact revision records.

The published model repository root contains merged FP16 weights. The unmerged adapter remains under `adapter/`, as documented in [`model_cards/retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md).
