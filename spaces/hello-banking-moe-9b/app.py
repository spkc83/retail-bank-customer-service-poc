"""Public ZeroGPU chat demo for the Hello Banking MoE checkpoint."""

from __future__ import annotations

from typing import Any

import gradio as gr
import torch
from policy import (
    OOD_RESPONSE,
    SENSITIVE_RESPONSE,
    generated_response_is_unsafe,
    is_sensitive,
)
from router import ROUTER_REVISION, LearnedBankingRouter, messages_for_route
from transformers import AutoModelForCausalLM, AutoTokenizer

import spaces

MODEL_ID = "spkc83/hello-banking-moe-9b"
MODEL_REVISION = "b2466ca4b157f420432a5e20a14573e83954deae"
GPU_PENDING_RESPONSE = (
    "The public interface is online, but GPU inference is not assigned yet. "
    "The Space owner must enable ZeroGPU before this banking prompt can run."
)
SYSTEM_PROMPT = (
    "You are a retail banking support assistant. Help only with accounts, cards, transfers, "
    "payments, loans, fees, branches, ATMs, and related financial-services support. You "
    "cannot access real accounts or perform transactions. Never request or accept a PIN, "
    "CVV/CVC, password, one-time code, or full card or account number. Direct users to their "
    "bank's official app, website, or verified phone number for account-specific help."
)

tokenizer = None
model = None
router = None
router_load_error = None
try:
    router = LearnedBankingRouter.from_hub()
except Exception as error:
    router_load_error = f"{type(error).__name__}: {error}"

if torch.cuda.is_available():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.output_router_logits = False
    model.eval()


def _gpu(**_kwargs: Any) -> Any:
    def decorator(function: Any) -> Any:
        return function

    return decorator


gpu = getattr(spaces, "GPU", _gpu)


def bounded_messages(
    message: str,
    history: list[dict[str, Any]],
    *,
    max_complete_turns: int = 4,
) -> list[dict[str, str]]:
    usable = [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in history
        if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str)
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *usable[-(max_complete_turns * 2) :],
        {"role": "user", "content": message},
    ]


@gpu(size="large", duration=120)
def generate_banking(message: str, history: list[dict[str, Any]]) -> str:
    if tokenizer is None or model is None:
        return GPU_PENDING_RESPONSE
    messages = bounded_messages(message, history)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    if generated_response_is_unsafe(response):
        return SENSITIVE_RESPONSE
    return response


def respond(message: str, history: list[dict[str, Any]]) -> str:
    if is_sensitive(message):
        return SENSITIVE_RESPONSE
    route = route_query(message, history)
    if route["route"] != "in_domain":
        return OOD_RESPONSE
    return generate_banking(message, history)


def route_query(
    message: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify a request with the learned banking domain and intent router."""
    if router is None:
        return fail_closed_route(f"router unavailable: {router_load_error}")
    try:
        result = router.classify(messages_for_route(message, history))
    except Exception as error:
        return fail_closed_route(f"invalid request or router failure: {type(error).__name__}")
    return {
        "route": result.route,
        "banking_probability": result.banking_probability,
        "intent": result.intent,
        "intent_confidence": result.intent_confidence,
        "confidence": result.confidence,
        "threshold": router.domain_threshold,
        "router_revision": ROUTER_REVISION,
        "reason": result.reason,
    }


def fail_closed_route(reason: str) -> dict[str, Any]:
    return {
        "route": "out_of_domain",
        "banking_probability": None,
        "intent": None,
        "intent_confidence": None,
        "confidence": 1.0,
        "threshold": router.domain_threshold if router is not None else None,
        "router_revision": ROUTER_REVISION,
        "reason": f"failed closed: {reason}",
    }


EXAMPLES = [
    "My debit card was stolen. What should I do?",
    "A transfer is still pending after two business days. What should I check?",
    "My card payment was declined at two different stores.",
    "Give me a recipe for chocolate cake.",
    "Please tell me my account balance and PIN.",
]

with gr.Blocks() as demo:
    gr.ChatInterface(
        fn=respond,
        type="messages",
        title="Hello Banking MoE 9B",
        description=(
            "Experimental public demo of a banking-focused 8.94B-parameter MoE checkpoint. "
            "A learned dual-head classifier routes the banking domain and 77 intents before "
            "generation. It cannot access real accounts or perform transactions. "
            "Do not enter sensitive data."
        ),
        examples=EXAMPLES,
        chatbot=gr.Chatbot(height=520, layout="bubble", type="messages"),
        textbox=gr.Textbox(
            placeholder="Ask about cards, accounts, transfers, payments, or loans…",
            max_length=1_000,
        ),
    )
    route_message = gr.Textbox(visible=False)
    route_history = gr.JSON(visible=False)
    route_output = gr.JSON(visible=False)
    route_button = gr.Button(visible=False)
    route_button.click(
        route_query,
        inputs=[route_message, route_history],
        outputs=route_output,
        api_name="route",
        api_description=(
            "Return the learned banking-domain probability and Banking77 intent. "
            "History is an optional list of role/content objects."
        ),
        queue=False,
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
