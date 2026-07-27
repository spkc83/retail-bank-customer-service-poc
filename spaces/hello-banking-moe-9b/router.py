"""Manifest-verified learned banking domain and intent router."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROUTER_REPO_ID = "spkc83/hello-banking-dual-head-router"
ROUTER_REVISION = "e7d928e5cf8c8be0883625f276c4e6c85c35eaf1"


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class RouteResult:
    route: Literal["in_domain", "out_of_domain"]
    confidence: float
    banking_probability: float
    intent: str | None
    intent_confidence: float
    reason: str


class DualHeadRouterModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_size: int,
        num_intents: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.domain_head = nn.Linear(hidden_size, 2)
        self.intent_head = nn.Linear(hidden_size, num_intents)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.domain_head(pooled), self.intent_head(pooled)


class LearnedBankingRouter:
    def __init__(
        self,
        *,
        tokenizer: Any,
        model: DualHeadRouterModel,
        intent_labels: Sequence[str],
        domain_threshold: float,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model.to("cpu").eval()
        self.intent_labels = tuple(intent_labels)
        self.domain_threshold = domain_threshold
        self.max_length = max_length

    @classmethod
    def from_hub(cls) -> LearnedBankingRouter:
        root = Path(
            snapshot_download(repo_id=ROUTER_REPO_ID, revision=ROUTER_REVISION)
        )
        config = verify_router_artifact(root)
        tokenizer = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        encoder = AutoModel.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        intent_labels = tuple(str(label) for label in config["intent_labels"])
        model = DualHeadRouterModel(
            encoder,
            hidden_size=int(encoder.config.hidden_size),
            num_intents=len(intent_labels),
        )
        heads = load_file(root / "classifier_heads.safetensors", device="cpu")
        model.domain_head.load_state_dict(
            {
                "weight": heads["domain_head.weight"],
                "bias": heads["domain_head.bias"],
            },
            strict=True,
        )
        model.intent_head.load_state_dict(
            {
                "weight": heads["intent_head.weight"],
                "bias": heads["intent_head.bias"],
            },
            strict=True,
        )
        return cls(
            tokenizer=tokenizer,
            model=model,
            intent_labels=intent_labels,
            domain_threshold=float(config["domain_threshold"]),
            max_length=int(config["max_length"]),
        )

    def classify(self, messages: Sequence[ChatMessage]) -> RouteResult:
        user_texts = [
            message.content.strip() for message in messages if message.role == "user"
        ]
        if not user_texts or not user_texts[-1]:
            raise ValueError("at least one non-empty user message is required")
        current_result = self._predict(f"[CURRENT]\n{user_texts[-1]}")
        if current_result.route == "in_domain":
            return current_result
        if len(user_texts) < 2 or not _has_contextual_reference(user_texts[-1]):
            return current_result
        return self._predict(
            f"[CURRENT]\n{user_texts[-1]}\n[PREVIOUS_USER]\n{user_texts[-2]}"
        )

    def _predict(self, rendered: str) -> RouteResult:
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {
            name: tensor.to("cpu")
            for name, tensor in encoded.items()
            if name in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode():
            domain_logits, intent_logits = self.model(**inputs)
        banking_probability = float(
            torch.softmax(domain_logits.float(), dim=-1)[0, 1]
        )
        intent_probabilities = torch.softmax(intent_logits.float(), dim=-1)[0]
        intent_confidence_tensor, intent_index_tensor = intent_probabilities.max(dim=-1)
        intent_confidence = float(intent_confidence_tensor)
        intent_index = int(intent_index_tensor)
        accepted = banking_probability >= self.domain_threshold
        return RouteResult(
            route="in_domain" if accepted else "out_of_domain",
            confidence=banking_probability if accepted else 1.0 - banking_probability,
            banking_probability=banking_probability,
            intent=self.intent_labels[intent_index] if accepted else None,
            intent_confidence=intent_confidence,
            reason=(
                f"learned domain probability {banking_probability:.6f} "
                f"{'>=' if accepted else '<'} threshold {self.domain_threshold:.6f}"
            ),
        )


def messages_for_route(
    message: str,
    history: list[dict[str, Any]] | None,
) -> list[ChatMessage]:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    if history is not None and not isinstance(history, list):
        raise ValueError("history must be a list of role/content objects")
    if any(not isinstance(item, dict) for item in (history or [])):
        raise ValueError("history entries must be role/content objects")
    messages = [
        ChatMessage(role=item["role"], content=item["content"])
        for item in (history or [])
        if item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]
    messages.append(ChatMessage(role="user", content=message.strip()))
    return messages


def verify_router_artifact(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != "banking-dual-head-router-artifact":
        raise ValueError("unexpected router artifact contract")
    if manifest.get("release_eligible") is not True:
        raise ValueError("router artifact did not pass release gates")
    for entry in manifest.get("files", []):
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe router artifact path: {relative}")
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing router artifact file: {relative}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"router artifact size mismatch: {relative}")
        if _file_sha256(path) != str(entry["sha256"]):
            raise ValueError(f"router artifact digest mismatch: {relative}")
    config = json.loads((root / "router_config.json").read_text(encoding="utf-8"))
    if config.get("contract") != "banking-dual-head-router":
        raise ValueError("unexpected router configuration contract")
    if config.get("fail_closed") is not True:
        raise ValueError("router artifact does not require fail-closed serving")
    return config


def _has_contextual_reference(text: str) -> bool:
    normalized = "".join(character.lower() if character.isalnum() else " " for character in text)
    terms = set(normalized.split())
    return bool(
        terms
        & {
            "again",
            "else",
            "it",
            "next",
            "that",
            "them",
            "then",
            "there",
            "these",
            "they",
            "this",
            "those",
        }
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
