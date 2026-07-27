---
title: Hello Banking MoE 9B
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
suggested_hardware: zero-a10g
models:
  - spkc83/retail-bank-servicing-moe-9b
  - spkc83/retail-bank-domain-intent-router
short_description: Guarded chat demo for an experimental banking MoE.
---

# Hello Banking MoE 9B demo

This public ZeroGPU Space runs the
[`spkc83/retail-bank-servicing-moe-9b`](https://huggingface.co/spkc83/retail-bank-servicing-moe-9b)
checkpoint for banking-support experiments.

The CPU frontend uses the release-gated
[`spkc83/retail-bank-domain-intent-router`](https://huggingface.co/spkc83/retail-bank-domain-intent-router)
for calibrated domain routing and Banking77 intent prediction. The artifact is
pinned by revision, hash-verified before loading, and fails closed if unavailable.
The public `route` API returns its banking probability and predicted intent.

ZeroGPU assignment remains pending because the current Hugging Face OAuth
credential cannot change Space hardware; accepted banking prompts report that
status until ZeroGPU is enabled.

The raw generative model is not production-safe. This demo applies learned
out-of-domain routing plus deterministic sensitive-data guards before
generation, and blocks generated requests for banking credentials. It cannot
access accounts or perform transactions.
