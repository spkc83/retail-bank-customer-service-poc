#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 SOURCE_COMMIT DATASET_REVISION SOURCE_MODEL_REVISION [MAX_STEPS]" >&2
  exit 2
fi

source_commit="$1"
dataset_revision="$2"
source_model_revision="$3"
max_steps="${4:-600}"
script_url="https://raw.githubusercontent.com/spkc83/retail-bank-servicing/${source_commit}/scripts/banking_v2/hf_job_continue_tool_sft.py"

if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_COMMIT must be the exact 40-character lowercase Git commit." >&2
  exit 2
fi

if [[ ! "$dataset_revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "DATASET_REVISION must be the exact 40-character lowercase Git commit." >&2
  exit 2
fi

if [[ ! "$source_model_revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_MODEL_REVISION must be the exact 40-character lowercase Git commit." >&2
  exit 2
fi

curl --fail --silent --show-error --head "$script_url" >/dev/null

job_args=(
  --flavor rtx-pro-6000
  --timeout 5h
  --secrets HF_TOKEN
  --volume hf://buckets/spkc83/jobs-artifacts:/data
  --label project=retail-bank-agent-v3-continuation
  --label source="${source_commit:0:8}"
  --label parent_model="${source_model_revision:0:8}"
  "$script_url"
  --source-commit "$source_commit"
  --dataset-revision "$dataset_revision"
  --source-model-revision "$source_model_revision"
  --output-dir "/data/retail-bank-agent-9b-continuation-${source_commit:0:8}-${source_model_revision:0:8}"
  --max-steps "$max_steps"
)

hf jobs uv run "${job_args[@]}"
