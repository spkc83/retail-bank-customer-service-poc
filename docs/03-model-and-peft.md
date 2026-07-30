# Model And PEFT

The active generative model is `spkc83/retail-bank-agent-9b`, a merged FP16
LoRA adaptation of IBM Granite for a synthetic retail-bank tool-use POC.

The source of truth for released identity and metrics is
[../model_cards/retail-bank-agent-9b.md](../model_cards/retail-bank-agent-9b.md).
The source of truth for local training defaults is
[../scripts/retail_bank/cloud_train_tool_sft.py](../scripts/retail_bank/cloud_train_tool_sft.py)
and [../configs/banking-tool-sft-granite.toml](../configs/banking-tool-sft-granite.toml).

## Base Model Identity

| Field | Value |
| --- | --- |
| Base model | `ibm-granite/granite-4.1-8b` |
| Base revision | `1504002f650e656a0a3789d99574df12e3e94ed0` |
| Architecture | Dense decoder-only causal transformer |
| Parameter count | 8,791,592,960 |
| Tool format | Granite native tagged JSON |
| Released model repo | `spkc83/retail-bank-agent-9b` |
| Immutable weights revision | `085df3d089cfadd77424b548542da0390a54a23e` |

The live POC loads that model repo and revision by default in
[../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py).

## PEFT Strategy

Training uses LoRA through PEFT and TRL SFTTrainer. The primary lane is BF16
LoRA over the pinned base model.

Defaults in [../scripts/retail_bank/cloud_train_tool_sft.py](../scripts/retail_bank/cloud_train_tool_sft.py):

| Setting | Value |
| --- | --- |
| Precision | `bf16-lora` |
| Optional precision | `qlora` |
| LoRA rank | `32` |
| LoRA alpha | `64` |
| LoRA dropout | `0.05` |
| Learning rate | `1e-4` |
| Max sequence length | `2048` |
| Training seed | `7303` |
| Default max steps | `1000` locally, `3000` in the HF job wrapper |
| Checkpoint interval | `250` locally, `500` in the HF job wrapper |

LoRA target modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

These names are declared in `LORA_TARGET_MODULES` in
[../scripts/retail_bank/cloud_train_tool_sft.py](../scripts/retail_bank/cloud_train_tool_sft.py)
and mirrored by the local TOML configuration.

## Training Record Rendering

Training examples are rendered by `ToolWireAdapter.render_training()` in
[../src/hello_slm/banking_tool_wire.py](../src/hello_slm/banking_tool_wire.py).
That adapter is responsible for:

- accepting only the Granite family;
- rendering tokenizer chat-template messages with tool schemas;
- preserving whole user-to-final-assistant tool chains inside the sequence
  budget;
- applying assistant-only labels;
- masking context, user messages, and tool results with `-100`;
- returning `input_ids`, `attention_mask`, `labels`, a span map, and the chat
  template hash.

The training worker pre-tokenizes records through `tokenize_records()` in
[../scripts/retail_bank/cloud_train_tool_sft.py](../scripts/retail_bank/cloud_train_tool_sft.py).

## Granite Tool Wire

The active tool wire is Granite-only. `_normalize_family()` in
[../src/hello_slm/banking_tool_wire.py](../src/hello_slm/banking_tool_wire.py)
raises an error for any non-Granite family.

Tool calls use tagged JSON blocks:

```text
<tool_call>
{"name":"freeze_card","arguments":{"last4":"4821"}}
</tool_call>
```

The parser validates:

- parseable JSON;
- object payloads;
- known public tool names;
- object arguments;
- allowed argument names;
- required arguments when a schema declares them;
- JSON value types and numeric bounds;
- unique and ordered call IDs/indexes.

The adapter intentionally does not infer intent, repair malformed output,
rename tools, or fill missing arguments. Invalid model output is a model
protocol error.

## Local Planning And Smoke Checks

The training worker is safe by default. Running it without remote execution
flags prints a dry-run plan and does not download the 8.79B base model, start a
paid job, merge weights, or push to Hugging Face:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
  --manifest data/banking-v3-tool-sft/manifest.json
```

The local tiny smoke path uses small offline stand-ins:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
  --run-tiny-smoke \
  --family granite \
  --max-steps 1 \
  --output-dir /tmp/banking-v3-tool-sft-smoke
```

Use the smoke path to prove tokenizer rendering, assistant-label masking,
checkpoint metadata, and tagged-JSON parsing without downloading the base
model.

## Remote Training Guard

Full remote execution requires all of these safeguards:

- `--execute-remote`
- `--allow-remote-execution`
- `RETAIL_BANK_ALLOW_REMOTE_TOOL_SFT=banking-v3-tool-sft`

The guarded wrapper is
[../scripts/retail_bank/run_remote_training_job.sh](../scripts/retail_bank/run_remote_training_job.sh).
It submits [../scripts/retail_bank/hf_job_tool_sft.py](../scripts/retail_bank/hf_job_tool_sft.py)
to Hugging Face Jobs with:

- exact source commit;
- exact dataset revision;
- `rtx-pro-6000` flavor;
- five-hour outer timeout;
- mounted artifact volume;
- `HF_TOKEN` as a secret;
- BF16 LoRA settings.

The job script downloads the pinned source archive, downloads the dataset
snapshot, then calls the guarded local worker with push-to-Hub enabled.

## Checkpoints And Fingerprints

`training_fingerprint()` in
[../scripts/retail_bank/cloud_train_tool_sft.py](../scripts/retail_bank/cloud_train_tool_sft.py)
captures:

- base model and revision;
- Granite family;
- tokenizer chat-template hash;
- dataset repository, revision, and manifest hash;
- training seed;
- precision;
- LoRA rank, alpha, dropout, and target modules.

`validate_resume_fingerprint()` rejects resume checkpoints whose metadata does
not match the current training inputs. This prevents accidental continuation
from a different base, dataset, template, precision, or adapter shape.

## Merge And Release Layout

The release keeps two forms:

- root checkpoint: merged FP16 weights;
- `adapter/`: retained unmerged LoRA adapter.

`merge_adapter_with_reload_parity()` in
[../scripts/retail_bank/cloud_train_tool_sft.py](../scripts/retail_bank/cloud_train_tool_sft.py)
merges the adapter, reloads the merged model, and compares adapter-vs-merged
outputs. The release helper
[../scripts/retail_bank/hf_job_finalize_tool_sft.py](../scripts/retail_bank/hf_job_finalize_tool_sft.py)
checks parity reports before publication.

The public model card reports the active release metrics:

| Metric | Value |
| --- | ---: |
| Representative parity generations | `8/8` exact |
| FP16 adapter/merged argmax agreement | `1.0` |
| Mean absolute logit drift | `0.00828770` |
| p99 absolute logit drift | `0.0410156` |
| p999 absolute logit drift | `0.0703125` |
| Maximum absolute logit drift | `0.238281` |

Merge parity is a release gate, not a replacement for frozen evaluation.

## Frozen Evaluation Summary

The model card records that the released checkpoint passed the frozen
1,347-record evaluation split with:

- `774/774` tool names and arguments;
- `678/678` executable tool trajectories;
- `96/96` exact dependent multi-tool sequences;
- `63/63` appropriate clarifications;
- `258/258` banking FAQ answers;
- `30/30` OOD response paths;
- `1,119/1,119` grounded factual responses;
- zero malformed calls, private arguments, credential requests, in-domain
  false refusals, or OOD false accepts.

The evaluator code is [../src/hello_slm/banking_tool_eval.py](../src/hello_slm/banking_tool_eval.py).
The remote evaluator entry points live under [../scripts/retail_bank](../scripts/retail_bank).

## Related Tests

Run the focused model/tool-wire tests from the repository root:

```bash
python -m pytest -q \
  tests/test_banking_tool_wire.py \
  tests/test_banking_tool_sft_worker.py \
  tests/test_banking_tool_sft_job.py \
  tests/test_banking_tool_sft_continuation.py \
  tests/test_banking_tool_sft_export_recovery.py \
  tests/test_banking_tool_eval.py \
  tests/test_banking_tool_eval_runner.py
```
