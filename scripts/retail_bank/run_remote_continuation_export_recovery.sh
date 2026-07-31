#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "usage: $0 RECOVERY_SOURCE_COMMIT TRAINING_SOURCE_COMMIT DATASET_REVISION" \
    "PARENT_MODEL_REVISION TRAINING_JOB OUTPUT_ROOT SELECTED_STEP" \
    "[SELECTED_ADAPTER_SUBDIR]" >&2
  exit 2
fi

recovery_source_commit="$1"
training_source_commit="$2"
dataset_revision="$3"
parent_model_revision="$4"
training_job="$5"
output_root="$6"
selected_step="$7"
selected_adapter_subdir="${8:-checkpoint-${selected_step}}"
script_url="https://raw.githubusercontent.com/spkc83/retail-bank-servicing/${recovery_source_commit}/scripts/retail_bank/hf_job_recover_continuation_export.py"

for revision in "$recovery_source_commit" "$training_source_commit" "$dataset_revision" "$parent_model_revision"; do
  if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "source, dataset, and parent revisions must be exact 40-character lowercase commits" >&2
    exit 2
  fi
done

if [[ "$output_root" != /data/retail-bank-agent-9b-continuation-* ]]; then
  echo "OUTPUT_ROOT must identify one continuation run under /data" >&2
  exit 2
fi

if [[ ! "$selected_step" =~ ^[1-9][0-9]*$ ]]; then
  echo "SELECTED_STEP must be a positive integer" >&2
  exit 2
fi

if ! curl --fail --silent --head "$script_url" >/dev/null 2>&1; then
  echo "Could not resolve bootstrap script: ${script_url}" >&2
  exit 2
fi

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
  --selected-adapter-subdir "$selected_adapter_subdir" \
  --selected-step "$selected_step" \
  --output-root "$output_root"
