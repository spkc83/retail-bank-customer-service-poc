from __future__ import annotations

import os
from typing import Any

import torch

from model_service import ModelDrivenBankingService
from policy import generated_response_is_unsafe
from state import BANK

MODEL_ID = "spkc83/hello-banking-moe-9b"
MODEL_REVISION = "b2466ca4b157f420432a5e20a14573e83954deae"
SKIP_MODEL_LOAD = os.environ.get("POC_SKIP_MODEL_LOAD") == "1"

if SKIP_MODEL_LOAD:

    class _Spaces:
        @staticmethod
        def GPU(**_kwargs: Any) -> Any:
            def decorator(function: Any) -> Any:
                return function

            return decorator

    spaces: Any = _Spaces()
    tokenizer = None
    model = None
else:
    import spaces
    from transformers import AutoModelForCausalLM, AutoTokenizer

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


@spaces.GPU(size="large", duration=90)
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
