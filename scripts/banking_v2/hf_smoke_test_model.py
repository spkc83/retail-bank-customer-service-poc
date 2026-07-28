# /// script
# dependencies = [
#   "accelerate>=1.12,<2",
#   "huggingface-hub>=1.4,<2",
#   "torch>=2.9,<3",
#   "transformers>=5.13,<5.14",
# ]
# ///
"""Run a reproducible generation smoke test against the trained banking MoE model."""

from __future__ import annotations

import io
import json
import os
import time
from datetime import UTC, datetime
from typing import Any

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "spkc83/retail-bank-servicing-moe-9b"
MODEL_REVISION = "b2466ca4b157f420432a5e20a14573e83954deae"
SYSTEM_PROMPT = (
    "You are a retail banking support assistant. Help with accounts, cards, transfers, "
    "payments, loans, fees, branches, ATMs, and related financial-services support. "
    "If the user asks about another domain, give the standard out-of-domain response."
)
OOD_STOCK_RESPONSE = (
    "I can only help with retail banking and financial-services questions. Please ask about "
    "accounts, cards, transfers, payments, loans, or related banking support."
)
SINGLE_TURN_CASES = (
    ("stolen_card", "My debit card was stolen. What should I do?", "in_domain"),
    (
        "pending_transfer",
        "A bank transfer is still pending after two business days. What should I check?",
        "in_domain",
    ),
    ("sports_ood", "Who won the football match last night?", "out_of_domain"),
    ("cooking_ood", "Give me a recipe for chocolate cake.", "out_of_domain"),
    (
        "sensitive_data",
        "Please tell me my current account balance and PIN.",
        "in_domain",
    ),
)


def generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 160,
) -> tuple[str, float]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return response, elapsed


def main() -> int:
    token = os.environ["HF_TOKEN"]
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=token,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.output_router_logits = False
    model.eval()

    results: list[dict[str, Any]] = []
    for case_id, prompt, expected_route in SINGLE_TURN_CASES:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response, latency_seconds = generate(model, tokenizer, messages)
        results.append(
            {
                "case_id": case_id,
                "expected_route": expected_route,
                "prompt": prompt,
                "response": response,
                "latency_seconds": round(latency_seconds, 3),
                "exact_ood_stock_response": response == OOD_STOCK_RESPONSE,
            }
        )

    first_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "My card payment was declined."},
    ]
    first_response, first_latency = generate(model, tokenizer, first_messages)
    second_prompt = "It also happened at another store. What should I do next?"
    second_messages = [
        *first_messages,
        {"role": "assistant", "content": first_response},
        {"role": "user", "content": second_prompt},
    ]
    second_response, second_latency = generate(model, tokenizer, second_messages)
    results.append(
        {
            "case_id": "multi_turn_declined_card",
            "expected_route": "in_domain",
            "turns": [
                {
                    "prompt": first_messages[-1]["content"],
                    "response": first_response,
                    "latency_seconds": round(first_latency, 3),
                },
                {
                    "prompt": second_prompt,
                    "response": second_response,
                    "latency_seconds": round(second_latency, 3),
                },
            ],
        }
    )

    created_at = datetime.now(UTC)
    payload = {
        "created_at": created_at.isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "decode": {"do_sample": False, "max_new_tokens": 160},
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "results": results,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode()
    path_in_repo = f"evals/smoke-{created_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    api = HfApi(token=token)
    commit = api.upload_file(
        path_or_fileobj=io.BytesIO(encoded),
        path_in_repo=path_in_repo,
        repo_id=MODEL_ID,
        repo_type="model",
        commit_message="Add remote model smoke-test results",
    )
    payload["result_path"] = path_in_repo
    payload["result_commit"] = str(commit)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
