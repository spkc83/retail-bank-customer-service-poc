# Frozen Two-Phase Evaluation

This guide covers the active frozen evaluation for the Granite PEFT model. Evaluation is read-only with respect to banking tools: it generates model outputs, appends canonical replay-validated tool results from the dataset when appropriate, scores the outputs, and publishes evaluation artifacts under the model repository.

## Active Artifact IDs

| Artifact | Value | Owner |
| --- | --- | --- |
| Model repo | `spkc83/retail-bank-agent-9b` | [`scripts/banking_v2/hf_job_tool_eval.py`](../scripts/banking_v2/hf_job_tool_eval.py) |
| Model revision | `085df3d089cfadd77424b548542da0390a54a23e` | [`model_cards/retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md) |
| Dataset repo | `spkc83/retail-bank-agent-sft` | [`scripts/banking_v2/hf_job_tool_eval.py`](../scripts/banking_v2/hf_job_tool_eval.py) |
| Dataset revision | `183e7e1ed1aba9c3d7155e7b83b64dc854935055` | [`data_cards/retail-bank-agent-sft.md`](../data_cards/retail-bank-agent-sft.md) |
| Frozen split | `test`, 1,347 records | [`data/banking-v3-tool-sft/preparation-report.json`](../data/banking-v3-tool-sft/preparation-report.json) |
| Evaluation job | `spkc83/6a6a6c7cb36a6516e96a0ac4` | [`model_cards/retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md) |
| Published evaluation path | `evaluation/085df3d089cf-183e7e1ed1ab/` | [`model_cards/retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md) |

## Evaluation Components

| Component | Purpose |
| --- | --- |
| [`scripts/banking_v2/evaluate_tool_model.py`](../scripts/banking_v2/evaluate_tool_model.py) | Local static evaluator CLI wrapper |
| [`src/hello_slm/banking_tool_eval.py`](../src/hello_slm/banking_tool_eval.py) | Metrics, parser adapter, dry-run fixture, report writer |
| [`scripts/banking_v2/cloud_generate_tool_eval.py`](../scripts/banking_v2/cloud_generate_tool_eval.py) | GPU prediction generator and scorer |
| [`scripts/banking_v2/hf_job_tool_eval.py`](../scripts/banking_v2/hf_job_tool_eval.py) | Pinned Hugging Face Jobs bootstrap |
| [`scripts/banking_v2/run_remote_tool_eval_job.sh`](../scripts/banking_v2/run_remote_tool_eval_job.sh) | Paid HF Jobs launcher |
| [`tests/test_banking_tool_eval.py`](../tests/test_banking_tool_eval.py) | Metric and parser scoring tests |
| [`tests/test_banking_tool_eval_runner.py`](../tests/test_banking_tool_eval_runner.py) | Two-phase generation, resume, and job wrapper tests |

## Two-Phase Contract

[`cloud_generate_tool_eval.py`](../scripts/banking_v2/cloud_generate_tool_eval.py) runs deterministic generation with the Granite tagged-JSON tool adapter.

Phase 1 asks the model for the first assistant response:

- For tool records, the prompt stops before the expected assistant tool-call message.
- For no-tool records, the prompt stops before the expected final assistant response.

If the model emits a tool call that exactly matches the next expected canonical call, the runner appends the dataset's canonical tool result and allows another model-owned pass. It never executes live tools. It stops on:

- no tool calls;
- unmatched tool call;
- missing canonical tool result;
- `--max-tool-passes`;
- `--max-tool-calls`.

Phase 2 asks for the grounded final response only after required tool records have canonical tool results appended. The metadata explicitly records:

- `tool_execution: false`;
- `deterministic_output_repair: false`;
- `teacher_forced_unseen_assistant_tool_calls: false`.

The runner resumes safely from an existing predictions JSONL and skips completed `record_id` values. Old phase-row prediction files are rejected.

## Local Dry Run

Use the static dry run to verify the scorer without GPU or Hub access:

```bash
PYTHONPATH=src python scripts/banking_v2/evaluate_tool_model.py \
  --dry-run \
  --checkpoint-revision local-docs-smoke \
  --output /tmp/hello-slm-tool-eval-dry-run.json
```

This command evaluates two in-memory records. It verifies tool name and argument scoring, executable tool success, grounded final factuality, OOD path scoring, credential request rate, and report serialization. It does not load the Granite model or the frozen 1,347-record split.

Run focused evaluation tests:

```bash
python -m pytest -q tests/test_banking_tool_eval.py \
  tests/test_banking_tool_eval_runner.py
```

## Full Local/GPU Runner

Use the generator directly only when the machine has appropriate GPU memory and Hub access:

```bash
PYTHONPATH=src python scripts/banking_v2/cloud_generate_tool_eval.py \
  --model-repo spkc83/retail-bank-agent-9b \
  --model-revision 085df3d089cfadd77424b548542da0390a54a23e \
  --dataset-repo spkc83/retail-bank-agent-sft \
  --dataset-revision 183e7e1ed1aba9c3d7155e7b83b64dc854935055 \
  --manifest data/banking-v3-tool-sft/manifest.json \
  --output-dir artifacts/tool-eval \
  --split test \
  --family granite \
  --device cuda \
  --dtype fp16 \
  --max-new-tokens-first 192 \
  --max-new-tokens-final 220 \
  --max-tool-passes 4 \
  --max-tool-calls 6
```

`--limit N` may be used for a local experiment, but `N` must be at least `1`. Do not use a limited run as a release gate.

Expected local outputs use the slug `{model_revision[:12]}-{dataset_revision[:12]}-{split}`:

- `predictions-085df3d089cf-183e7e1ed1ab-test.jsonl`;
- `metadata-085df3d089cf-183e7e1ed1ab-test.json`;
- `report-085df3d089cf-183e7e1ed1ab-test.json`.

## Paid HF Jobs Evaluation

The paid launcher is [`scripts/banking_v2/run_remote_tool_eval_job.sh`](../scripts/banking_v2/run_remote_tool_eval_job.sh). It validates exact source, model, and dataset revisions, checks that the pinned bootstrap URL exists, then launches:

- `hf jobs uv run`;
- `--flavor rtx-pro-6000`;
- `--timeout 2h`;
- `--secrets HF_TOKEN`;
- `--volume hf://buckets/spkc83/jobs-artifacts:/data`;
- labels `project=retail-bank-agent-v3-eval` and `model=${MODEL_REVISION:0:8}`.

```bash
bash scripts/banking_v2/run_remote_tool_eval_job.sh \
  SOURCE_COMMIT_40_HEX \
  085df3d089cfadd77424b548542da0390a54a23e \
  183e7e1ed1aba9c3d7155e7b83b64dc854935055
```

The `HF_TOKEN` secret must read the model and dataset and write evaluation artifacts to `spkc83/retail-bank-agent-9b`. The launcher writes to `/data/retail-bank-agent-eval-${MODEL_REVISION:0:8}-${DATASET_REVISION:0:8}` in the durable bucket.

[`hf_job_tool_eval.py`](../scripts/banking_v2/hf_job_tool_eval.py) pins the runtime packages in its PEP 723 header, downloads the exact source commit, sets `PYTHONPATH`, exports revision metadata, and invokes [`cloud_generate_tool_eval.py`](../scripts/banking_v2/cloud_generate_tool_eval.py) with `--dtype fp16 --push-to-hub`.

## Published Outputs

When `--push-to-hub` is set, [`publish_eval_artifacts`](../scripts/banking_v2/cloud_generate_tool_eval.py) uploads predictions, metadata, and report to:

```text
evaluation/{model_revision[:12]}-{dataset_revision[:12]}/
```

For the released model this is:

```text
evaluation/085df3d089cf-183e7e1ed1ab/
```

The metadata records:

- model repo and revision;
- dataset repo and revision;
- manifest path and SHA-256;
- public tool manifest hash;
- split name;
- decoding limits;
- first-assistant and grounded-final record counts;
- output paths and SHA-256 values;
- publish prefix;
- disabled live tool execution and disabled output repair.

## Metrics and Gates

[`evaluate_records`](../src/hello_slm/banking_tool_eval.py) reports numerators, denominators, and scores for:

| Metric | Release rule |
| --- | --- |
| Tool-name accuracy | exact ordered names over expected calls |
| Tool-argument accuracy | exact schema-normalized arguments with names and order |
| Executable tool success | generated public calls match replay-validated expectations |
| Multi-tool exact sequence | exact count, order, names, and args |
| Clarification appropriateness | asks only for the expected missing field |
| No-tool FAQ quality | includes required facts without forbidden contradictions |
| OOD/small-talk path | no banking tool call and expected scope response |
| Grounded final factuality | includes required facts and avoids forbidden facts |
| Malformed tool-call rate | attempted call is unparseable |
| Unsupported/private arguments | generated args fail public manifest |
| Credential request rate | asks for account/customer/password/PIN-like secrets |
| In-domain false refusal | supported turn wrongly refused |
| OOD false accept | OOD turn generated banking tool calls |

The release gates are enforced by the scorer and runner tests around [`src/hello_slm/banking_tool_eval.py`](../src/hello_slm/banking_tool_eval.py) and [`scripts/banking_v2/cloud_generate_tool_eval.py`](../scripts/banking_v2/cloud_generate_tool_eval.py). A candidate cannot pass by averaged score if a hard gate fails.

The released frozen evaluation passed:

| Slice | Result |
| --- | ---: |
| Tool names and arguments | `774/774` |
| Executable tool trajectories | `678/678` |
| Exact dependent multi-tool sequences | `96/96` |
| Appropriate clarifications | `63/63` |
| Banking FAQ answers | `258/258` |
| OOD response paths | `30/30` |
| Grounded factual responses | `1,119/1,119` |
| Malformed calls | `0` |
| Unsupported/private arguments | `0` |
| Credential requests | `0` |
| In-domain false refusals | `0` |
| OOD false accepts | `0` |

## Stop Conditions

Stop evaluation and do not publish a release claim if:

- any revision is not an exact 40-character lowercase Git commit;
- the manifest is unavailable or does not declare the requested split;
- prediction JSONL is corrupt or uses the old phase-row contract;
- the model emits unmatched required tool calls and metrics fall below gate;
- `max_tool_passes` or `max_tool_calls` truncates required calls;
- the report lacks dataset fingerprint, adapter template hash, or checkpoint revision;
- any hard gate fails;
- the paid job cannot persist to `/data` or upload evaluation artifacts.

Evaluation does not repair model output and does not execute live tools. A failed evaluation should produce a failed report, not a corrected prediction file.
