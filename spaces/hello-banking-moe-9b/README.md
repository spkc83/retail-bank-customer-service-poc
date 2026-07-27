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
  - spkc83/hello-banking-moe-9b
short_description: Guarded chat demo for an experimental banking MoE.
---

# Hello Banking MoE 9B demo

This public ZeroGPU Space runs the
[`spkc83/hello-banking-moe-9b`](https://huggingface.co/spkc83/hello-banking-moe-9b)
checkpoint for banking-support experiments.

Deployment status: the public CPU frontend is live. ZeroGPU assignment is
pending because the current Hugging Face OAuth credential cannot change Space
hardware; in-domain generation reports that status until ZeroGPU is enabled.

The raw model is not production-safe. This demo applies deterministic
out-of-domain and sensitive-data guards before generation and blocks generated
requests for banking credentials. It cannot access accounts or perform
transactions.
