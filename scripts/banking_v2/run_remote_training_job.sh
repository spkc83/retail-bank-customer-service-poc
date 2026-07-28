#!/usr/bin/env bash
set -euo pipefail

cd /workspace
uv run --extra scale scripts/banking_v2/cloud_train_banking_moe.py \
  --execute-remote \
  --allow-remote-execution \
  --push-to-hub \
  --hub-dest spkc83/hello-banking-moe-9b \
  --manifest data/banking-v2/manifest.json \
  --output-dir /tmp/hello-slm-banking-v2-artifacts \
  --max-steps 1000 \
  --batch-size 1 \
  --max-seq-len 512 \
  --learning-rate 2e-5 \
  --checkpoint-every 250
