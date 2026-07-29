---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: transformers
pipeline_tag: text-generation
tags:
  - banking
  - conversational
  - mixture-of-experts
  - qwen2-moe
widget:
  - text: "Show my account balances."
  - text: "My debit card was stolen. What should I do?"
  - text: "Why was my card payment declined?"
---

# Retail Bank Servicing MoE 9B

Retail Bank Servicing MoE 9B is an experimental retail-banking support model. It expands
the language representations of `Qwen/Qwen2.5-1.5B-Instruct` into a
Qwen2-MoE checkpoint and adapts the routed residual experts and routers on a
restricted banking-support corpus.

Source, training, evaluation, and serving code:
https://github.com/spkc83/retail-bank-servicing

This is a research and demonstration checkpoint, not a production banking
assistant. It cannot access accounts, balances, cards, PINs, transactions, or
bank systems. Do not provide passwords, PINs, account numbers, or other secrets.

## Architecture

- `Qwen2MoeForCausalLM`
- 8,943,713,792 total parameters
- approximately 2,073,443,840 active parameters per token
- 28 layers and 28 routed experts per layer
- top-2 routed experts per token
- BF16 weights
- 151,936-token inherited vocabulary

The model is not a 9B model trained from random initialization. Compatible
language, attention, embedding, and normalization weights were copied from the
pinned 1.5B base checkpoint. The MoE routed branches were initialized to
preserve the dense model's initial output, then trained for 1,000 optimizer
steps.

## Training data

Generative adaptation used:

- 22,033 training conversations from the Bitext retail-banking corpus and
  self-authored OOD/multi-turn examples;
- 1,008 validation conversations;
- 1,001 held-out test conversations.

The generative splits contain CDLA-Sharing-1.0 and MIT material. Banking77
(CC-BY-4.0) is reserved for intent-router evaluation and was not used for
generative training.

## Preliminary training results

- final training loss: 1.2638
- validation loss: 0.7775
- every per-layer expert-health gate passed at steps 250, 500, 750, and 1,000

These training metrics do not establish production quality. Broader held-out
response, multi-turn, hallucination, safety, and calibrated domain-router
evaluations remain required.

### Raw generation smoke test

A deterministic six-scenario smoke test on an RTX PRO 6000 found that the model
produced fluent banking responses and carried a declined-card scenario into a
second turn. It also found release-blocking behavior:

- neither raw OOD generation returned the required stock response;
- a cooking prompt received a recipe instead of a refusal;
- a sensitive account/PIN prompt invited the user to provide account details;
- a declined-card response requested card number, expiration date, and CVV.

The raw result is stored in
`evals/smoke-20260727T122519Z.json`. Do not expose the checkpoint without
external domain and sensitive-data controls.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "spkc83/retail-bank-servicing-moe-9b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map={"": 0},
)

# Router logits and auxiliary loss are training diagnostics. Disable them for
# generation with the current Transformers Qwen2-MoE implementation.
model.config.output_router_logits = False
model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "You are a retail banking support assistant. Help with accounts, cards, "
            "transfers, payments, loans, fees, branches, ATMs, and related "
            "financial-services support. If the user asks about another domain, "
            "give the standard out-of-domain response."
        ),
    },
    {"role": "user", "content": "My debit card was stolen. What should I do?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
with torch.inference_mode():
    output = model.generate(inputs, max_new_tokens=160, do_sample=False)
print(tokenizer.decode(output[0, inputs.shape[-1] :], skip_special_tokens=True))
```

## Domain boundary

The public application uses a separate calibrated domain/intent router.
High-confidence OOD requests bypass neural generation. Allowed and uncertain
turns reach this model with the intent head's top-three predictions as advisory
context. The 9B model, rather than a capability planner, owns conversation,
tool selection, tool arguments, and final wording. Explicit high-confidence
non-banking requests return:

> I can only help with retail banking and financial-services questions. Please
> ask about accounts, cards, transfers, payments, loans, or related banking
> support.

Prompting or fine-tuning alone does not guarantee that exact response. The
released DistilBERT router has a binary supported-banking/OOD head and a 77-way
Banking77 intent head. Its held-out intent macro F1 is `0.948425`, with OOD
false-accept rate `0.020109` at its calibrated `0.165` lower boundary. The
public POC is an
experimental synthetic environment, not a production-qualified banking agent.

## Historical public POC serving role

This section records the earlier MoE experiment. The current public
[Retail Bank Servicing POC](https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc)
uses `spkc83/retail-bank-agent-9b`, not this checkpoint.

1. Static Gradio authentication identifies one of two synthetic demo users.
2. One directly registered ZeroGPU event owns the complete turn.
3. Its CPU-resident dual-head router gates high-confidence OOD; ranked intent
   predictions are diagnostics. Accepted and uncertain turns continue to the
   9B model, which responds directly or emits one or more Qwen tool calls.
4. A session-isolated SQLite backend executes generated calls against synthetic
   records.
5. Tool results return to the model for a customer-facing grounded generation.

The application retains complete session history and selects newest complete
conversation/tool interactions within an 8,192-token input budget. It does not
replace model responses with deterministic grounded templates. Per-pass prompt
and output hashes, raw outputs, call counts, and runtime device metadata
distinguish direct and grounded-final generations.

## Limitations

- May provide incorrect, incomplete, or unsafe financial guidance.
- Cannot authenticate users or perform banking actions.
- May hallucinate bank policies, fees, timelines, or contact information.
- Restricted training coverage limits linguistic and scenario diversity.
- The OOD and intent classifier meets the POC gates but is not production-qualified.
- Public demo presets are smoke tests, not proof of generalization.

## Deployment status

This MoE checkpoint is retained as an evaluation control and is not the active
public POC model. The active deployment uses the dense Granite-based
`spkc83/retail-bank-agent-9b` checkpoint for accepted and uncertain
conversation, tool calling, and final response generation.

The deployment uses eager expert execution for compatibility with the current
ZeroGPU partition. If ZeroGPU is unavailable, the POC reports model
unavailability and does not substitute a CPU-generated banking response.

Use the model only for experimentation with human review.
