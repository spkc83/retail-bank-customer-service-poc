from __future__ import annotations

import os
from typing import Any

from model_service import SYSTEM_PROMPT, TOOL_SCHEMAS, ModelDrivenBankingService
from policy import generated_response_is_unsafe
from state import BANK

MODEL_ID = "spkc83/retail-bank-servicing-moe-9b"
MODEL_REVISION = "b2466ca4b157f420432a5e20a14573e83954deae"
SKIP_MODEL_LOAD = os.environ.get("POC_SKIP_MODEL_LOAD") == "1"

if SKIP_MODEL_LOAD:
    import torch

    tokenizer = None
    model = None
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        experts_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.config.output_router_logits = False
    model.eval()


class TransformersGenerator:
    def generate(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]] | None,
        max_new_tokens: int,
    ) -> str:
        if tokenizer is None or model is None:
            raise RuntimeError("ZeroGPU model is unavailable")
        template_arguments: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools is not None:
            template_arguments["tools"] = tools
        rendered = tokenizer.apply_chat_template(messages, **template_arguments)
        encoded = tokenizer(rendered, return_tensors="pt")
        inputs = {name: tensor.to(model.device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        return str(tokenizer.decode(new_ids, skip_special_tokens=True)).strip()


def gpu_allocation_probe() -> dict[str, Any]:
    """Verify that ZeroGPU entered application code with an initialized CUDA device."""

    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name": str(torch.cuda.get_device_name(0)),
    }


def inspect_model_selection(message: str) -> str:
    """Return the model's raw first-pass tool selection for authenticated diagnostics."""

    return TransformersGenerator().generate(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        tools=TOOL_SCHEMAS,
        max_new_tokens=128,
    )


def inspect_model_service(message: str) -> str:
    """Run the full model/tool path in an isolated synthetic diagnostic session."""

    try:
        result = run_model_service("alex.demo", "model-service-probe", message, [])
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return str(result)


def run_model_service(
    username: str,
    session_hash: str,
    message: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    service = ModelDrivenBankingService(
        bank=BANK,
        generator=TransformersGenerator(),
    )
    reply = service.reply(
        username=username,
        session_hash=session_hash,
        message=message,
        history=history,
    )
    if generated_response_is_unsafe(reply.response):
        raise RuntimeError("model response requested prohibited credentials")
    return {
        "response": reply.response,
        "tool_name": reply.tool_name,
        "tool_result": reply.tool_result,
        "snapshot": reply.snapshot,
        "model_revision": MODEL_REVISION,
    }
