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
  `085df3d089cfadd77424b548542da0390a54a23e`
- Training/provenance head:
  `247ac402989144698f89727a59a07ce5d05f31c6`
- Base revision:
  `1504002f650e656a0a3789d99574df12e3e94ed0`
- Source revision:
  `4270636255515f7a563d935794a3642e0b13ccb3`
- Recovery source revision:
  `0237b97c0a9558bbb2e95c45097ac5ae5f9f7f21`
- Training job:
  `spkc83/6a6a60d4b36a6516e96a0709`
- FP16-native recovery and merge-parity job:
  `spkc83/6a6a6b6323ed89c748ec502c`
- Dataset revision:
  `183e7e1ed1aba9c3d7155e7b83b64dc854935055`
- Dataset fingerprint:
  `2bb7a400ed2556b15c7e5eb6147668041b5deef8ae4f037f9e2e52295ff29ab5`
- Parameters: 8,791,592,960
- Architecture: dense decoder-only causal transformer
- Tool format: Granite native tagged JSON

## Adaptation

- 6,304 training conversations
- 1,349 validation conversations
- 1,347 frozen test conversations
- BF16 LoRA over attention and MLP projection modules
- LoRA rank 32, alpha 64, dropout 0.05
- 2,048-token maximum training sequence
- corrective continuation learning rate `5e-5`
- effective batch size 4
- FP16-native merged root checkpoint
- unmerged adapter retained under `adapter/`

## Training and merge results

- 600-step servicing-quality continuation in 1,011.198 seconds
- aggregate continuation training loss: `0.0601915`
- final validation loss: `0.000095959`
- final validation token accuracy: `1.0`
- eight representative 32-token parity generations: `8/8` exact
- FP16 adapter/merged argmax agreement: `1.0`
- mean absolute logit drift: `0.00828770`
- p99 / p999 absolute logit drift: `0.0410156` / `0.0703125`
- maximum absolute logit drift: `0.238281`

The final FP32-accumulated merge was rejected at the original decimal p999
boundary. The release uses an FP16-native merge and a quantization-aligned
p999 ceiling of `0.0703125`; all generated parity outputs remained identical.
The full frozen generation evaluation remains the release-quality gate; logit
parity is not presented as bitwise equality.

## Frozen evaluation

Evaluation job `spkc83/6a6a6c7cb36a6516e96a0ac4` decoded the 1,347-record
frozen split on CUDA with deterministic FP16 generation. The report is stored
under `evaluation/085df3d089cf-183e7e1ed1ab/` in the model repository.

- tool names and arguments: `774/774`
- executable tool trajectories: `678/678`
- exact dependent multi-tool sequences: `96/96`
- appropriate clarifications: `63/63`
- banking FAQ answers: `258/258`
- OOD response paths: `30/30`
- grounded factual responses: `1,119/1,119`
- malformed calls, unsupported/private arguments, credential requests,
  in-domain false refusals, and OOD false accepts: `0`

All 43 held-out account-balance cases included the requested monetary facts.
All 48 held-out mortgage-age cases stated the typical United States minimum of
18 and retained lender/jurisdiction and eligibility caveats.

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
