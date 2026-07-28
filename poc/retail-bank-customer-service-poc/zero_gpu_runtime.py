from __future__ import annotations

import json
import os
from typing import Any

MODEL_ID = "spkc83/retail-bank-servicing-moe-9b"
MODEL_REVISION = "b2466ca4b157f420432a5e20a14573e83954deae"
SKIP_MODEL_LOAD = os.environ.get("POC_SKIP_MODEL_LOAD") == "1"

if SKIP_MODEL_LOAD:

    class _Spaces:
        @staticmethod
        def GPU(**_kwargs: Any) -> Any:
            def decorator(function: Any) -> Any:
                return function

            return decorator

    spaces_runtime: Any = _Spaces()
    tokenizer = None
    model = None
else:
    import spaces as spaces_runtime
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


@spaces_runtime.GPU(size="large", duration=90)
def generate_final_answer(
    messages: list[dict[str, str]],
    grounded_results: dict[str, Any],
    max_new_tokens: int,
) -> str:
    """Run only stateless grounded response generation on ZeroGPU."""

    if tokenizer is None or model is None:
        raise RuntimeError("ZeroGPU model is unavailable")
    if not messages or messages[0].get("role") != "system":
        raise ValueError("finalizer messages must begin with a system prompt")
    if not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be between 1 and 512")

    rendered_messages = [dict(item) for item in messages]
    rendered_messages[0]["content"] = (
        f"{rendered_messages[0]['content']}\n\n"
        "VERIFIED SYNTHETIC WORKFLOW RESULTS (the only factual source):\n"
        f"{json.dumps(grounded_results, sort_keys=True)}"
    )
    rendered = tokenizer.apply_chat_template(
        rendered_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
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
