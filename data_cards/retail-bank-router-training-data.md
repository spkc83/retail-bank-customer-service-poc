---
license: cc-by-4.0
task_categories:
  - text-classification
language:
  - en
---

# Retail Bank router training data

Governed classifier-only data derived from PolyAI Banking77 and UCI CLINC150.
It is not included in generative SFT.

Preparation and audit code:
https://github.com/spkc83/retail-bank-model-development

Trained dual-head router:
https://huggingface.co/spkc83/retail-bank-domain-intent-router

- Train rows: 44,832
- Validation rows: 8,669
- Test rows: 16,380
- Domain labels: OOD=0, supported retail banking=1
- Intent labels: 77 Banking77 intents; `-100` means no intent supervision
- Licenses: CC-BY-4.0

See `manifest.json` in the dataset repository for source revisions, hashes,
mapping policy, and audit counts.
