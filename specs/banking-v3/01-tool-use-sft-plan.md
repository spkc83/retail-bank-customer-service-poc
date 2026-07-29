# Banking v3 tool-use SFT plan

Status: implementation contract on branch `feat/tool-use-sft-v3`.

The user authorized end-to-end implementation, public Hub publication, a
single RTX PRO 6000 training job capped at five hours, and deployment to the
existing public ZeroGPU demonstration.

## Outcome

Build a model-driven retail-banking assistant that:

- remains approximately 9B total parameters;
- selects and calls the nine existing banking tools;
- uses tool results to produce grounded final responses;
- supports clarification, errors, no-tool banking questions, OOD turns, and
  multi-turn conversations;
- trains on one Hugging Face Jobs RTX PRO 6000 with a five-hour cap;
- serves as a merged checkpoint through the existing ZeroGPU application.

The current 8.943B custom Qwen2-MoE remains an evaluation control. It is not the
default v3 architecture because its learned language and agentic capability
originates in a 1.5B base and the existing SFT corpus contains no structured
tool trajectories.

## Decisions

1. Replace prompt-only repair with governed tool-use SFT.
2. Prefer a natively pretrained model in the 8.75B-9.25B release band.
3. Use BF16 LoRA on the actual 96GB RTX PRO 6000 Jobs flavor. Keep QLoRA as an
   explicit fallback. Do not use full-parameter Adam on one GPU.
4. Merge the selected adapter into the base before ZeroGPU deployment.
5. Generate semantic scenarios deterministically and use a teacher LLM only
   for linguistic realization.
6. Generalize the mechanical tool wire adapter per model family. Do not force
   every candidate to emit Qwen XML.
7. Keep tool selection, arguments, clarification, and final response generation
   model-owned. The runtime may parse, validate, execute, and return tool
   results, but it may not infer intent or repair a call.
8. Keep Banking77 and CLINC classifier/evaluation-only unless a separate
   governance decision changes their role.

## Data-generation strategy

### Why not a VAE or GAN

Tool-use training needs exact conditional sequences, typed arguments, backend
state transitions, and factually entailed final answers. A VAE or GAN makes
these properties harder to control and validate. The generator therefore uses
a symbolic scenario engine for semantics and an autoregressive teacher only
for language variation.

### Generation pipeline

1. Freeze the nine-tool public manifest and mock-bank behavior.
2. Create a deterministic initial customer and bank state.
3. Select a scenario family and outcome.
4. Produce the expected ordered tool plan.
5. Replay every call against an isolated mock-bank session.
6. Store exact results and the expected final-state hash.
7. Ask a teacher model to realize the user and assistant wording without
   changing calls, arguments, results, or facts.
8. Run schema, tool-manifest, replay, grounding, PII, provenance,
   chat-template, and split-leakage validators.
9. Accept, quarantine for rewrite, or reject the record.

Teacher output never decides whether a tool is required. It cannot rename a
tool, add an argument, modify a result, or claim an action succeeded.

### Scenario families

- Read success: accounts, cards, service cases, transactions, and transfers.
- Write success: freeze card, replace card, dispute transaction, and cancel
  transfer.
- Ordered and repeated multi-tool calls.
- Clarification for ambiguous cards, merchants, and transfers.
- Backend and business-state errors, including already-completed actions.
- In-domain no-tool FAQs grounded in a synthetic bank knowledge base.
- Greetings, thanks, unrelated queries, and OOD transitions.
- Hard negatives: asking for account numbers, customer IDs, passwords, PINs,
  private backend IDs, or claiming an unexecuted action succeeded.

### Scale and mixture

The validated corpus contains 5,000 conversations: 3,502 train, 748
validation, and 750 test. The original 35,000-60,000 aspirational range was
rejected after the generator's duplicate-text gate proved that scaling beyond
the natural realization space would add repeated prompts. The accepted corpus
contains about 4.25 million rendered tokens; three-plus training epochs remain
inside the bounded run while preserving unique normalized user turns.

| Slice | Target share |
| --- | ---: |
| Successful single- and multi-tool trajectories | 60%-70% |
| No-tool banking FAQ and conversational continuity | 10%-15% |
| OOD, greeting, thanks, and use-original behavior | 10%-15% |
| Clarification, backend errors, and hard negatives | 10%-15% |
| Audited and rewritten Bitext examples | At most 10%-15% |

Bitext rows that request account numbers or identity data are excluded unless
rewritten and revalidated. Existing generic QA must not dominate the tool-use
mixture.

### Split isolation

Assign splits by the tuple of scenario family, initial state seed, synthetic
customer, template, and realization seed. None of these units may cross
training, validation, or test splits. Keep a separate live-regression set
containing POC presets and all previously observed failure prompts.

## Canonical tool-use record

The stored representation is independent of any model family.

```json
{
  "schema_version": "banking-tool-sft/v1",
  "record_id": "freeze_001_realization_02",
  "messages": [
    {
      "role": "user",
      "content": "Freeze my debit card ending in 4821.",
      "loss": false
    },
    {
      "role": "assistant",
      "content": null,
      "loss": true,
      "tool_calls": [
        {
          "id": "call_freeze_001_realization_02_0",
          "index": 0,
          "type": "function",
          "function": {
            "name": "freeze_card",
            "arguments": {"last4": "4821"}
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_freeze_001_realization_02_0",
      "name": "freeze_card",
      "content": {
        "ok": true,
        "result": {
          "card": {
            "last4": "4821",
            "status": "frozen"
          },
          "simulated": true
        }
      },
      "loss": false
    },
    {
      "role": "assistant",
      "content": "Your debit card ending in 4821 is now frozen.",
      "loss": true
    }
  ],
  "expected": {
    "requires_tool": true,
    "ordered_calls": ["call_freeze_001_realization_02_0"],
    "final_state_hash": "sha256:...",
    "grounding_facts": ["card.last4=4821", "card.status=frozen"]
  },
  "split_keys": {
    "scenario_family": "card_freeze",
    "state_seed": "state-014",
    "customer_id": "synthetic-customer-014",
    "template_id": "freeze-explicit-v1",
    "realization_seed": "realization-002"
  },
  "provenance": {
    "source": "self-authored-synthetic",
    "license": "MIT",
    "generator_version": "banking-tool-sft/v1",
    "teacher_model": "model-id",
    "teacher_prompt_hash": "sha256:..."
  },
  "validation": {
    "tool_manifest_hash": "sha256:...",
    "replay_hash": "sha256:...",
    "accepted": true
  }
}
```

Canonical arguments are typed JSON values. Tool results use
`{"ok": true, "result": <exact backend result>}` or
`{"ok": false, "error": {"code": <stable code>, "message": <safe text>}}`.
The runtime creates this envelope mechanically and does not repair a call.
Family adapters may render these values as strings when required by a tokenizer
template. Every tool call has a stable ID and index so repeated calls to the
same tool remain correlated.

Only assistant tool-call tokens and assistant final-response tokens receive
labels. System, user, and tool-result tokens are context only. Packing and
truncation must preserve a complete user-to-final-assistant tool chain.

## Tool wire adapter

Each supported family implements the same syntax-only contract:

```text
render_tools(public_tool_manifest) -> family tool definitions
render_training(messages) -> input_ids, labels, canonical span map
render_generation(messages, tools) -> model inputs
parse_assistant(generated_tokens) -> canonical assistant message
render_tool_result(canonical_tool_message) -> family tool-result message
```

The adapter may decode tokens, template syntax, and JSON. It may not rename a
tool, insert or rename an argument, infer a missing value, choose a fallback
tool, or replace invalid output. Parsed arguments must validate against the
public manifest before execution. Training, evaluation, and serving use the
same adapter and template hash.

Golden fixtures for Qwen, Granite, Qwen3.5, and optional Ministral must prove:

- assistant-only labels;
- stable call IDs and repeated-call correlation;
- no labels on tool results;
- whole-chain packing and truncation;
- parseable generation after adapter save, reload, and merge.

## Base-model and architecture bakeoff

| Priority | Candidate | Parameters | Strength | Principal risk |
| ---: | --- | ---: | --- | --- |
| 1 | `ibm-granite/granite-4.1-8b` | 8.792B | Apache-2.0, dense CausalLM, explicit function calling, low integration risk | Template and parser differ from current Qwen runtime |
| 2 | `Qwen/Qwen3.5-9B` | 9B language component | Literal 9B, Apache-2.0, strong agent benchmarks | Multimodal wrapper, hybrid language architecture, newer stack |
| 3 | `Qwen/Qwen3-8B` | 8.19B | Mature Apache Qwen tool path and simplest migration | Outside the preferred 8.75B-9.25B release band |
| 4 | `mistralai/Ministral-3-8B-Instruct-2512` | 8.918B total | Apache-2.0 and native function calling | 8.4B language plus 0.4B vision and newer VLM stack |
| Control | Current custom Qwen2-MoE | 8.944B | Existing trained and deployed artifact | Only 1.5B pretrained language core; no tool SFT |

Granite is the provisional default because it is inside the release band and
has the simplest text CausalLM path. Qwen3.5 is the literal-9B challenger.
Qwen3 is the fallback only if all release-band candidates fail engineering
gates.

Every candidate must pass:

1. License and exact parameter-count verification.
2. Pinned Transformers, PEFT, TRL, Accelerate, and CUDA compatibility.
3. Public tool-template round trip.
4. Zero-shot held-out tool evaluation.
5. One Trainer/tokenization optimizer smoke and an in-job BF16 LoRA startup
   gate.
6. Adapter save/reload and merged-checkpoint reload.
7. Text-only load and memory proof for multimodal-wrapper candidates.
8. ZeroGPU model/tool round trip through the POC.
9. Throughput projection with at least 20% margin under the five-hour cap.

A failed QLoRA, merge/reload, or ZeroGPU gate disqualifies the candidate from a
paid full run.

## Training plan

Primary stack:

- Transformers and Accelerate;
- TRL `SFTTrainer`;
- PEFT BF16 LoRA;
- optional `bitsandbytes` 4-bit NF4 QLoRA fallback.

The candidate smoke resolves and pins exact package versions, CUDA version,
base/tokenizer revisions, and template hash. Full jobs may not depend on an
unpinned development branch.

Starting ranges:

| Setting | Pilot range |
| --- | --- |
| LoRA rank | 16, 32, 64; start full candidate at 32 |
| LoRA alpha | Twice the rank |
| LoRA dropout | 0.03-0.10; start at 0.05 |
| Target modules | Attention and MLP projections |
| Sequence length | 2,048-4,096 |
| Micro-batch | 1-2 |
| Effective batch | 4 through accumulation |
| Learning rate | 5e-5 to 2e-4; start at 1e-4 |
| Warmup | 3%-5% |
| Epochs | Approximately 3.4 at the 3,000-step ceiling |
| Checkpoints | Every 250-500 optimizer steps |

The Jobs hardware inventory reports 96GB VRAM for `rtx-pro-6000`, so BF16 LoRA
is the lower-risk primary lane. QLoRA remains available through an explicit
precision switch. Full-parameter Adam training is out of scope.

The final artifact includes:

- the adapter checkpoint;
- the FP32-accumulated, merged FP16 checkpoint;
- tokenizer and tool template;
- base, data, package, and template fingerprints;
- resume state and evaluation report.

## Token and five-hour budget

Measured corpus budget:

1. Granite template hash:
   `6727ca16a39df05c41af54eb651aa618b50a29967ad3951a31b90c4e385573fc`.
2. Training split: 2,974,599 input tokens and 144,442 labeled assistant tokens.
3. Validation split: 636,374 input tokens and 30,821 labeled assistant tokens.
4. Test split: 636,648 input tokens and 31,180 labeled assistant tokens.
5. The worker stops optimizer work after 14,400 seconds even if the 3,000-step
   ceiling is not reached.
6. The outer Hugging Face Job timeout remains five hours, leaving one hour for
   startup, validation, fresh-base merge, reload parity, and Hub upload.

## Evaluation contract

| Metric | Frozen denominator | Scoring rule | Gate |
| --- | ---: | --- | ---: |
| Banking77 classifier intent macro F1 | Existing test split | Standard macro F1 | >= 0.90 |
| In-domain false refusal | 570 tool-bearing records | Refusal path / all supported turns | <= 2% |
| OOD false accept | 57 OOD records | Banking tool call / all OOD turns | <= 1% |
| Tool-name accuracy | 609 expected calls | Exact name and order / all expected calls | >= 0.95 |
| Tool-argument accuracy | 609 expected calls | Exact schema-normalized args, name, and order / all expected calls | >= 0.90 |
| Executable tool success | 570 tool-bearing records | Exact replay-validated public call / all tool scenarios | >= 0.93 |
| Multi-tool exact sequence | 39 multi-tool records | Exact count, order, names, and args / all scenarios | >= 0.85 |
| Clarification appropriateness | 48 ambiguous records | Requests only expected missing field / all ambiguous turns | >= 0.85 |
| Grounded final factuality | 750 records | All required facts and no critical contradiction / all finals | >= 0.95 |
| Malformed tool-call rate | 750 decisions | Unparseable attempted calls / all decisions | < 1% |
| Unsupported/private arguments | All generated calls | Manifest failures / all calls | < 0.5% |
| Credential request rate | 750 records | Account/customer/password/PIN requests / all turns | 0 |
| No-tool FAQ quality | 36 FAQ records | Required facts without contradiction / all FAQs | >= 0.90 |
| OOD response path | 57 OOD records | Expected no-tool scope response / all turns | >= 0.95 |

Missing, malformed, or extra calls count as failures. The deterministic harness
owns parsing, manifest, replay, state, path, and structured-fact scores.
The automated frozen report is the reproducible release gate. A blinded human
review of clarification, factuality, and FAQ outputs is recommended before any
claim beyond this research POC; an LLM judge may triage but is not the sole
release scorer.

Every report records numerator, denominator, parse failures, dataset
fingerprint, adapter/template hash, and checkpoint revision. A candidate that
fails any hard gate cannot win through an averaged score.

## Implementation phases

1. Freeze tool, scenario, data, and evaluation contracts.
2. Implement the canonical record schema and validators.
3. Implement deterministic scenario planning and mock replay.
4. Add constrained teacher realization and quarantine reports.
5. Implement family wire adapters and assistant-only tokenization.
6. Run the local/low-cost candidate bakeoff.
7. Generate only the throughput-supported full corpus.
8. Run the guarded full job with startup, wall-clock, checkpoint, and parity
   gates.
9. Evaluate the resulting checkpoint and retain it only if the frozen metrics
   improve on the control.
10. Merge, validate, and deploy only a checkpoint that passes all gates.

Likely implementation files:

- `src/hello_slm/banking_tool_sft_data.py`
- `scripts/banking_v2/prepare_tool_sft_data.py`
- `scripts/banking_v2/cloud_train_tool_sft.py`
- `configs/banking-tool-sft-*.toml`
- family adapter modules shared by training and the POC;
- schema, replay, split, serializer, QLoRA, model-service, and ZeroGPU tests;
- dataset and model cards.

## Stop rules

- Do not start the paid run before data replay, adapter/tokenizer, exact TRL
  construction, one-step offline Trainer, job packaging, and Hub inputs pass.
- Stop a pilot on unsupported/private arguments, credential-request regression,
  adapter merge parity failure, or projected runtime beyond the cap.
- Stop if tool validation worsens at two consecutive checkpoints.
- Do not use a deterministic planner or output-repair fallback to make a model
  pass.

## Architecture decision record

Decision: replace the custom expanded MoE as the primary architecture with a
natively pretrained approximately-9B base, teach tool use through governed SFT,
train through QLoRA/LoRA, and merge the adapter for ZeroGPU.

Drivers:

- no structured tool trajectories in the current corpus;
- repeated invalid account-number behavior under base and reflection prompts;
- model-owned orchestration requirement;
- one 96GB RTX PRO 6000 and five-hour training cap;
- ordinary Transformers/ZeroGPU serving is lower risk than the custom MoE.

Alternatives rejected:

- Keep the expanded MoE as primary: useful only as a control because new expert
  capacity did not create pretrained language or agentic knowledge.
- Full-parameter fine-tuning: optimizer state and activations do not fit the
  current memory/cost envelope reliably.
- Deterministic planner plus smaller generator: violates the experiment.
- VAE/GAN synthesis: weak controllability for exact stateful tool trajectories.
- Literal 9B only: unnecessarily rejects lower-risk checkpoints that round to
  the same 9B deployment class.

Consequences:

- more data and adapter engineering precedes training;
- base selection remains conditional on measured bakeoff evidence;
- model-family syntax is isolated behind one shared adapter contract;
- a standard merged base should simplify serving compared with the custom MoE.

Follow-up: record the winning base and final parameter tier in a second ADR
after the bakeoff.

## Implementation staffing and verification

Recommended lanes after plan approval:

- dependency review: model revisions, licenses, templates, and package pins;
- data implementation: schema, scenarios, realization, and validators;
- training implementation: family adapters, assistant-only labels, QLoRA, and
  merge/export;
- test engineering: golden fixtures, replay, leakage, metrics, and smokes;
- independent verification: candidate evidence, budget, and ZeroGPU proof;
- documentation: dataset card, model card, and final ADR.

The local stop condition is passing unit tests, static checks, an exact
TRL/PEFT optimizer smoke, save/reload, and model-service tests. The remote stop
condition is a completed capped job, merged-checkpoint parity, frozen behavior
evaluation, and a proven public ZeroGPU deployment.

## Official references

- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Granite 4.1-8b model card](https://huggingface.co/ibm-granite/granite-4.1-8b)
- [Ministral 3 8B Instruct model card](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
- [Hugging Face PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
- [Hugging Face ZeroGPU documentation](https://huggingface.co/docs/hub/en/spaces-zerogpu)
