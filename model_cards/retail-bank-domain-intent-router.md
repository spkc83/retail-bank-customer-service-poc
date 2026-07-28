---
base_model: distilbert/distilbert-base-uncased
datasets:
  - spkc83/retail-bank-router-training-data
library_name: transformers
license: apache-2.0
pipeline_tag: text-classification
tags:
  - banking
  - intent-classification
  - out-of-domain-detection
---

# Retail Bank domain-intent router

DistilBERT shared encoder with a binary supported-banking/OOD head and a
77-way Banking77 intent head. The intent loss is masked for CLINC rows.

Source and evaluation code:
https://github.com/spkc83/retail-bank-servicing

Live model-driven application:
https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc

## Held-out results

- Release eligible: `True`
- Intent macro F1: `0.951208`
- In-domain false-refusal rate: `0.013689`
- OOD false-accept rate: `0.007733`
- Follow-up false-refusal rate: `0.006818`
- Banking-to-OOD false-accept rate: `0.005085`
- Calibrated banking threshold: `0.980000`

## Data and licenses

Classifier-only data combines PolyAI Banking77 and UCI CLINC150 under
CC-BY-4.0. Banking77 is prohibited from the generative SFT lane. The prepared
dataset contains 44,832 training, 8,669 validation, and 16,380 test rows.

## Safety boundary

If the artifact is unavailable or corrupt, serving must fail closed to the
standard OOD response. This model does not replace credential-input or
unsafe-output guards.
