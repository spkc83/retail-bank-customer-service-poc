#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 SOURCE_COMMIT DATASET_REVISION [RESUME_FROM]" >&2
  exit 2
fi

source_commit="$1"
dataset_revision="$2"
resume_from="${3:-}"
script_url="https://raw.githubusercontent.com/spkc83/retail-bank-servicing/${source_commit}/scripts/banking_v2/hf_job_tool_sft.py"

if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_COMMIT must be the exact 40-character lowercase Git commit." >&2
  exit 2
fi

curl --fail --silent --show-error --head "$script_url" >/dev/null

job_args=(
  --flavor rtx-pro-6000
  --timeout 5h
  --secrets HF_TOKEN
  --volume hf://buckets/spkc83/jobs-artifacts:/data
  --label project=retail-bank-agent-v3
  --label source="${source_commit:0:8}"
  "$script_url"
  --source-commit "$source_commit"
  --dataset-revision "$dataset_revision"
  --output-dir "/data/retail-bank-agent-9b-${source_commit:0:8}"
)

if [[ -n "$resume_from" ]]; then
  job_args+=(--resume-from "$resume_from")
fi

hf jobs uv run "${job_args[@]}"
