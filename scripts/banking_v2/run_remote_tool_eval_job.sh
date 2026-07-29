#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SOURCE_COMMIT MODEL_REVISION DATASET_REVISION" >&2
  exit 2
fi

source_commit="$1"
model_revision="$2"
dataset_revision="$3"
script_url="https://raw.githubusercontent.com/spkc83/retail-bank-servicing/${source_commit}/scripts/banking_v2/hf_job_tool_eval.py"

for revision_name in source_commit model_revision dataset_revision; do
  revision_value="${!revision_name}"
  if [[ ! "$revision_value" =~ ^[0-9a-f]{40}$ ]]; then
    echo "${revision_name} must be an exact 40-character lowercase Git commit." >&2
    exit 2
  fi
done

curl --fail --silent --show-error --head "$script_url" >/dev/null

hf jobs uv run \
  --flavor rtx-pro-6000 \
  --timeout 2h \
  --secrets HF_TOKEN \
  --volume hf://buckets/spkc83/jobs-artifacts:/data \
  --label project=retail-bank-agent-v3-eval \
  --label model="${model_revision:0:8}" \
  "$script_url" \
  --source-commit "$source_commit" \
  --model-revision "$model_revision" \
  --dataset-revision "$dataset_revision" \
  --output-dir "/data/retail-bank-agent-eval-${model_revision:0:8}-${dataset_revision:0:8}"
