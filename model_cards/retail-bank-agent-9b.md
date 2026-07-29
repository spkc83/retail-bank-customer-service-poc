---
license: apache-2.0
base_model: ibm-granite/granite-4.1-8b
datasets:
- spkc83/retail-bank-agent-sft
pipeline_tag: text-generation
tags:
- retail-banking
- tool-calling
- conversational
- peft
---

# Retail Bank Agent 9B

Retail Bank Agent 9B is an experimental customer-service and tool-use model for
the linked synthetic retail-bank demonstration. It is a merged BF16 LoRA
adaptation of `ibm-granite/granite-4.1-8b`.

- Source: https://github.com/spkc83/retail-bank-servicing
- Training dataset:
  https://huggingface.co/datasets/spkc83/retail-bank-agent-sft
- Public ZeroGPU POC:
  https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc

## Artifact identity

- Model repository: `spkc83/retail-bank-agent-9b`
- Released model revision: `PIN_AFTER_TRAINING`
- Base revision:
  `1504002f650e656a0a3789d99574df12e3e94ed0`
- Source revision:
  `2996273af060504c7bba913691228cc8687b67fb`
- Training job:
  `spkc83/6a6a113eb36a6516e969f750`
- Dataset revision:
  `fcf065dbb524f387d456f731dd708fba6da0f361`
- Dataset fingerprint:
  `bea295674348caffc3561474635c8af8b55836041c81296b46940456610e01af`
- Parameters: 8,791,592,960
- Architecture: dense decoder-only causal transformer
- Tool format: Granite native tagged JSON

## Adaptation

- 3,502 training conversations
- 748 validation conversations
- 750 frozen test conversations
- BF16 LoRA over attention and MLP projection modules
- LoRA rank 32, alpha 64, dropout 0.05
- 2,048-token maximum training sequence
- learning rate `1e-4`
- effective batch size 4
- merged BF16 root checkpoint
- unmerged adapter retained under `adapter/`

Final optimizer, validation, merge-parity, and release-evaluation values are
filled from the completed training artifact before release.

## Intended use

The model is intended only for research evaluation in the linked synthetic
banking POC. It must receive the published tool schemas, conversation history,
and correlated tool results. The model does not connect to real banking
systems.

## Limitations

The dataset is synthetic and deliberately narrow. The model may choose the
wrong tool, produce invalid arguments, mishandle conversation context, or make
unsupported claims. It is not financial advice and must not receive
credentials, full account numbers, payment-card details, or real customer data.

Release evaluation covers tool-call syntax and names, public arguments,
multi-tool ordering, clarification, tool errors, grounded final responses,
multi-turn follow-ups, OOD behavior, and malformed-call handling.
