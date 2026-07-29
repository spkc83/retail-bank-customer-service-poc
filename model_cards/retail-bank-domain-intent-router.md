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
- Model revision: `136ee159d19cda7f585dd122907bbeb1ef4ec4db`
- Training-data revision: `54ff186a03501d76dc643dbed3d82729267ce811`
- Intent macro F1: `0.948425`
- In-domain false-refusal rate: `0.005099`
- OOD false-accept rate: `0.020109`
- Follow-up false-refusal rate: `0.001623`
- Conversational false-refusal rate: `0.050000`
- Banking-to-OOD false-accept rate: `0.009783`
- Calibrated lower boundary: `0.165000`

The hosted POC uses two serving boundaries: banking probability below `0.165`
is OOD, probability at least `0.50` is in-domain, and the middle region is
uncertain. Uncertain turns continue to the 9B agent. The top three intent
predictions are displayed only as diagnostics; they do not enter the prompt,
select a tool, or provide tool arguments.

## Data and licenses

Classifier-only data combines PolyAI Banking77 and UCI CLINC150 under
CC-BY-4.0. Banking77 is prohibited from the generative SFT lane. The prepared
dataset contains 44,432 training, 8,589 validation, and 16,260 test rows.
Supported greeting, thanks, goodbye, and bot-identity examples are positive
domain rows with their 77-way intent loss masked.

## Serving fallback

If the artifact is unavailable or classification fails, the experimental POC
marks the route uncertain and delegates the turn to the 9B model. The router is
an experiment component, not a production authorization or safety boundary.
