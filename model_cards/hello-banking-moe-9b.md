---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: transformers
pipeline_tag: text-generation
tags:
  - banking
  - conversational
  - mixture-of-experts
  - qwen2-moe
---

# Hello Banking MoE 9B

Hello Banking MoE 9B is an experimental retail-banking support model. It expands
the language representations of `Qwen/Qwen2.5-1.5B-Instruct` into a
Qwen2-MoE checkpoint and adapts the routed residual experts and routers on a
restricted banking-support corpus.

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

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "spkc83/hello-banking-moe-9b"
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

The intended application uses a separate calibrated domain/intent router. An
out-of-domain decision bypasses neural generation and returns:

> I can only help with retail banking and financial-services questions. Please
> ask about accounts, cards, transfers, payments, loans, or related banking
> support.

Prompting or fine-tuning alone does not guarantee that exact response. The
public demo includes a deterministic prototype gate; production use requires a
trained and calibrated router evaluated on held-out banking and non-financial
prompts.

## Limitations

- May provide incorrect, incomplete, or unsafe financial guidance.
- Cannot authenticate users or perform banking actions.
- May hallucinate bank policies, fees, timelines, or contact information.
- Restricted training coverage limits linguistic and scenario diversity.
- The current OOD and intent classifier is not yet production-qualified.
- Public demo presets are smoke tests, not proof of generalization.

Use the model only for experimentation with human review.
