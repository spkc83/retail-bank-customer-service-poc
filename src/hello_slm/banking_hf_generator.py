from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hello_slm.banking_policy import ChatMessage, DomainRouteResult

BANKING_MODEL_ENV = "RETAIL_BANK_MODEL"
DEFAULT_BANKING_MODEL_PATH = Path("artifacts") / "banking-v2-moe-9b" / "final"
BANKING_SYSTEM_PROMPT = (
    "You are a retail banking support assistant. "
    "Answer only questions about accounts, cards, transfers, payments, loans, "
    "or related financial-services support. Do not answer unrelated questions."
)


@dataclass(frozen=True)
class HFGenerationSettings:
    max_new_tokens: int = 160
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.08


class MissingBankingCheckpointError(RuntimeError):
    """Raised when in-domain banking inference is requested before a model exists."""


class HuggingFaceBankingGenerator:
    """Lazy Transformers-backed generator for the trained banking checkpoint."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        settings: HFGenerationSettings | None = None,
        device: str | None = None,
    ) -> None:
        if model_path is None:
            configured_path: str | Path = os.environ.get(
                BANKING_MODEL_ENV, str(DEFAULT_BANKING_MODEL_PATH)
            )
        else:
            configured_path = model_path
        self.model_path = Path(configured_path)
        self.settings = settings or HFGenerationSettings()
        self.requested_device = device
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: torch.device | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        seed: int,
        route: DomainRouteResult,
    ) -> str:
        del route
        self._ensure_loaded()
        if self._tokenizer is None or self._model is None or self._device is None:
            raise RuntimeError("banking generator failed to initialize")

        rendered_messages = _to_transformers_messages(messages)
        input_ids = self._tokenizer.apply_chat_template(
            rendered_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._device)
        generator = torch.Generator(device=self._device).manual_seed(seed)
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=self.settings.temperature > 0,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                repetition_penalty=self.settings.repetition_penalty,
                pad_token_id=self._pad_token_id(),
                eos_token_id=getattr(self._tokenizer, "eos_token_id", None),
                generator=generator,
            )
        new_ids = output_ids[0, input_ids.shape[-1] :]
        return str(self._tokenizer.decode(new_ids, skip_special_tokens=True)).strip()

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        resolved = self.model_path.resolve()
        if not resolved.exists():
            raise MissingBankingCheckpointError(
                f"trained banking checkpoint is missing: {resolved}. "
                f"Set {BANKING_MODEL_ENV} to a local Transformers checkpoint directory."
            )
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise MissingBankingCheckpointError(
                "Transformers is required for banking-v2 inference but is not installed."
            ) from exc

        requested_device = self.requested_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        device = torch.device(requested_device)
        tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        model.config.output_router_logits = False
        model.to(device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _pad_token_id(self) -> int | None:
        if self._tokenizer is None:
            return None
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        return int(eos_token_id) if eos_token_id is not None else None


def _to_transformers_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    rendered: list[dict[str, str]] = [{"role": "system", "content": BANKING_SYSTEM_PROMPT}]
    for message in messages:
        if message.role == "system":
            rendered[0] = {"role": "system", "content": message.content}
            continue
        rendered.append({"role": message.role, "content": message.content})
    return rendered
