---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - banking
  - tool-calling
  - synthetic
  - conversational
pretty_name: Retail Bank Agent Tool-Use SFT
---

# Retail Bank Agent Tool-Use SFT

This dataset contains 9,000 deterministic, fictional retail-banking
conversations for supervised fine-tuning of a conversational tool-using model.

- Dataset: https://huggingface.co/datasets/spkc83/retail-bank-agent-sft
- Training revision:
  `183e7e1ed1aba9c3d7155e7b83b64dc854935055`
- Source: https://github.com/spkc83/retail-bank-servicing
- Model: https://huggingface.co/spkc83/retail-bank-agent-9b
- Public POC:
  https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc

## Splits

- Train: 6,304
- Validation: 1,349
- Frozen test: 1,347
- Corpus fingerprint:
  `2bb7a400ed2556b15c7e5eb6147668041b5deef8ae4f037f9e2e52295ff29ab5`
- Split seed: `711`

## Coverage

The corpus covers all nine public synthetic-bank tools, successful and failed
tool results, clarification, general banking FAQ, hard-negative private-field
requests, out-of-domain refusal, multi-turn context, and ordered multi-tool
calls.

| Scenario family | Conversations |
|---|---:|
| Clarification | 333 |
| Conversation | 999 |
| Hard negative | 333 |
| Multi-turn | 1,665 |
| No-tool banking FAQ | 1,665 |
| OOD | 333 |
| Tool error | 666 |
| Tool success | 3,006 |

Every tool-bearing record was replayed against isolated deterministic
synthetic state before inclusion. Assistant tool-call and final-response tokens
are trainable; system, user, and tool-result tokens are context only.
The data validator rejects semantically empty final responses and asserts
path-specific content for clarification, FAQ, OOD, and hard-negative rows.

## Source and privacy policy

All included rows are self-authored synthetic data under MIT. External
classifier corpora are prepared by a separate pipeline and never enter the
generative SFT splits.

The dataset contains no real customers, credentials, accounts, or financial
events. It is for research demonstrations, not production banking.
