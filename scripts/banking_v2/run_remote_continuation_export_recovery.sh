#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 RECOVERY_SOURCE_COMMIT TRAINING_SOURCE_COMMIT DATASET_REVISION PARENT_MODEL_REVISION TRAINING_JOB" >&2
  exit 2
fi

recovery_source_commit="$1"
training_source_commit="$2"
dataset_revision="$3"
parent_model_revision="$4"
training_job="$5"
script_url="https://raw.githubusercontent.com/spkc83/retail-bank-servicing/${recovery_source_commit}/scripts/banking_v2/hf_job_recover_continuation_export.py"
output_root="/data/retail-bank-agent-9b-continuation-68e96a7d-00c4ba1b"

for revision in "$recovery_source_commit" "$training_source_commit" "$dataset_revision" "$parent_model_revision"; do
  if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "source, dataset, and parent revisions must be exact 40-character lowercase commits" >&2
    exit 2
  fi
done

curl --fail --silent --show-error --head "$script_url" >/dev/null

hf jobs uv run \
  --flavor rtx-pro-6000 \
  --timeout 1h \
  --secrets HF_TOKEN \
  --volume hf://buckets/spkc83/jobs-artifacts:/data \
  --label project=retail-bank-agent-v3-export-recovery \
  --label source="${recovery_source_commit:0:8}" \
  "$script_url" \
  --recovery-source-commit "$recovery_source_commit" \
  --training-source-commit "$training_source_commit" \
  --dataset-revision "$dataset_revision" \
  --parent-model-revision "$parent_model_revision" \
  --training-job "$training_job" \
  --selected-adapter-subdir "checkpoint-500" \
  --selected-step 500 \
  --output-root "$output_root"
