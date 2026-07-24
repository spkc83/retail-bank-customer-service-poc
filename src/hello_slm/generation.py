from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from hello_slm.artifacts import sha256_file
from hello_slm.config import ExperimentConfig
from hello_slm.data import normalize_text
from hello_slm.tokenizer import SPECIAL_TOKENS, TokenizerError, load_tokenizer
from hello_slm.training import PipelineError, load_model_from_checkpoint

DEFAULT_SYSTEM = (
    "You are Hello SLM, a small local chat model trained from a restricted corpus. "
    "Answer only within the corpus and task instructions. If the answer is unsupported, "
    "say you do not know."
)


def generate_reply(
    config: ExperimentConfig,
    *,
    checkpoint_path: str | Path,
    prompt: str,
    as_json: bool = False,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    started = time.time()
    normalized = normalize_text(prompt)
    if not normalized.strip():
        raise PipelineError("empty user message")
    _reject_disallowed_characters(config, normalized)
    device = torch.device(config.data["training"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model, _, _ = load_model_from_checkpoint(config, checkpoint_path, device=device)
    tokenizer_path = config.artifact_dir / "tokenizer" / "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    rendered = _render_prompt(config, normalized)
    try:
        prompt_ids = tokenizer.encode(rendered, allow_unk=False)
    except TokenizerError as exc:
        raise PipelineError(f"prompt cannot be encoded by restricted tokenizer: {exc}") from exc
    max_context = min(int(config.data["model"]["max_seq_len"]), _context_default(config))
    if len(prompt_ids) > max_context:
        raise PipelineError("rendered prompt exceeds context limit")

    generation = config.data["generation"]
    requested_tokens = (
        int(generation["max_new_tokens"]) if max_new_tokens is None else int(max_new_tokens)
    )
    if requested_tokens < 1 or requested_tokens > 256:
        raise PipelineError("max_new_tokens must be between 1 and 256")
    available_tokens = model.config.max_seq_len - len(prompt_ids)
    if available_tokens < 1:
        raise PipelineError("rendered prompt leaves no room for generation")
    max_new_tokens = min(requested_tokens, available_tokens)
    top_k = int(generation["top_k"]) or None
    if top_k is not None:
        top_k = min(top_k, int(config.data["model"]["vocab_size"]))
    generator = torch.Generator(device=device).manual_seed(int(config.data["seeds"]["generation"]))
    output = model.generate(
        torch.tensor(prompt_ids, dtype=torch.long, device=device),
        max_new_tokens=max_new_tokens,
        temperature=float(generation["temperature"]),
        top_k=top_k,
        top_p=float(generation["top_p"]),
        repetition_penalty=float(generation["repetition_penalty"]),
        stop_ids={SPECIAL_TOKENS["<|end|>"], SPECIAL_TOKENS["<|eos|>"]},
        generator=generator,
    )
    ids = output[0].detach().cpu().tolist()
    new_ids = ids[len(prompt_ids) :]
    stop_reason = "max_new_tokens"
    if new_ids and new_ids[-1] == SPECIAL_TOKENS["<|end|>"]:
        stop_reason = "end_token"
    elif new_ids and new_ids[-1] == SPECIAL_TOKENS["<|eos|>"]:
        stop_reason = "eos_token"
    visible_ids = [
        token_id
        for token_id in new_ids
        if token_id not in {SPECIAL_TOKENS["<|end|>"], SPECIAL_TOKENS["<|eos|>"]}
    ]
    text = tokenizer.decode(visible_ids, skip_special_tokens=True).strip()
    result = {
        "response": text,
        "metadata": {
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(new_ids),
            "stop_reason": stop_reason,
            "latency_seconds": time.time() - started,
            "decoding": {
                "max_new_tokens": max_new_tokens,
                "temperature": float(generation["temperature"]),
                "top_k": top_k,
                "top_p": float(generation["top_p"]),
                "repetition_penalty": float(generation["repetition_penalty"]),
            },
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "json": as_json,
        },
    }
    return result


def _render_prompt(config: ExperimentConfig, user_message: str) -> str:
    if config.data["run"]["id"] in {
        "smoke",
        "arithmetic-30m",
        "arithmetic-curriculum-30m",
    }:
        return f"<|bos|><|user|>\n{user_message}<|end|>\n<|assistant|>\n"
    return (
        f"<|bos|><|system|>\n{DEFAULT_SYSTEM}<|end|>\n"
        f"<|user|>\n{user_message}<|end|>\n<|assistant|>\n"
    )


def _reject_disallowed_characters(config: ExperimentConfig, text: str) -> None:
    allowed = set(str(config.data["tokenizer"].get("allowed_characters", "")).replace("\\n", "\n"))
    invalid = sorted({char for char in text if char not in allowed})
    if invalid:
        raise PipelineError(f"prompt contains disallowed characters: {invalid!r}")


def _context_default(config: ExperimentConfig) -> int:
    return 128 if config.data["run"]["id"] == "smoke" else 1024
