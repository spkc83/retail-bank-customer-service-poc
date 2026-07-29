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

This dataset contains 5,000 deterministic, fictional retail-banking
conversations for supervised fine-tuning of a conversational tool-using model.

- Dataset: https://huggingface.co/datasets/spkc83/retail-bank-agent-sft
- Training revision:
  `fcf065dbb524f387d456f731dd708fba6da0f361`
- Source: https://github.com/spkc83/retail-bank-servicing
- Model: https://huggingface.co/spkc83/retail-bank-agent-9b
- Public POC:
  https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc

## Splits

- Train: 3,502
- Validation: 748
- Frozen test: 750
- Corpus fingerprint:
  `bea295674348caffc3561474635c8af8b55836041c81296b46940456610e01af`
- Split seed: `7303`

## Coverage

The corpus covers all nine public synthetic-bank tools, successful and failed
tool results, clarification, general banking FAQ, hard-negative private-field
requests, out-of-domain refusal, multi-turn context, and ordered multi-tool
calls.

| Scenario family | Conversations |
|---|---:|
| Clarification | 294 |
| Hard negative | 294 |
| Multi-turn | 588 |
| No-tool banking FAQ | 294 |
| OOD | 294 |
| Tool error | 588 |
| Tool success | 2,648 |

Every tool-bearing record was replayed against isolated deterministic
synthetic state before inclusion. Assistant tool-call and final-response tokens
are trainable; system, user, and tool-result tokens are context only.

## Source and privacy policy

All included rows are self-authored synthetic data under MIT. Banking77 and
CLINC are classifier/evaluation-only and contribute no generative SFT rows.
Bitext remains quarantined and contributes no rows.

The dataset contains no real customers, credentials, accounts, or financial
events. It is for research demonstrations, not production banking.
