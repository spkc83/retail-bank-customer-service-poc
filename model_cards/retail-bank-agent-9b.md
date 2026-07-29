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
the linked synthetic retail-bank demonstration. It is a merged FP16 LoRA
adaptation of `ibm-granite/granite-4.1-8b`.

- Source: https://github.com/spkc83/retail-bank-servicing
- Training dataset:
  https://huggingface.co/datasets/spkc83/retail-bank-agent-sft
- Public ZeroGPU POC:
  https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc

## Artifact identity

- Model repository: `spkc83/retail-bank-agent-9b`
- Immutable weights revision:
  `32f327ba162ef8988255017694dd6b8983d3af34`
- Training/provenance head:
  `53e4d50367b0013c3ad47d3404f04c46fa27570e`
- Base revision:
  `1504002f650e656a0a3789d99574df12e3e94ed0`
- Source revision:
  `3a6a7efe22b9ea2a104712cbeff5648df3eeec31`
- Training job:
  `spkc83/6a6a19a1b36a6516e969f78b`
- FP32-to-FP16 remerge job:
  `spkc83/6a6a2b4f23ed89c748ec3b2a`
- Merge-parity job:
  `spkc83/6a6a2be323ed89c748ec3b36`
- Dataset revision:
  `c0e0be08f9d56f382e3c85a6bca1e4f4090eacac`
- Dataset fingerprint:
  `d8014d6e7eda0d30f403461395c17882719fbe6b5b2c8f1ad4fe44deb25cd270`
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
- FP32-accumulated, merged FP16 root checkpoint
- unmerged adapter retained under `adapter/`

## Training and merge results

- 3,000 optimizer steps in 3,570.523 seconds
- aggregate training loss: `0.0507329`
- final validation loss: `0.0184957`
- final validation token accuracy: `0.996323`
- eight representative 32-token parity generations: `8/8` exact
- FP16 adapter/merged argmax agreement: `1.0`
- mean absolute logit drift: `0.00833845`
- p99 / p999 absolute logit drift: `0.0390625` / `0.0644531`
- maximum absolute logit drift: `0.28125`

The initial direct BF16 merge was rejected because it changed one of eight
representative generations and only reached `0.972763` argmax agreement. The
released weights were therefore merged in FP32 and cast to FP16. The full
frozen generation evaluation remains the release-quality gate; logit parity is
not presented as bitwise equality.

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
