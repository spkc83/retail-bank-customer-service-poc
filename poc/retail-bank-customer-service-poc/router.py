from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROUTER_REPO_ID = "spkc83/retail-bank-domain-intent-router"
ROUTER_REVISION = "136ee159d19cda7f585dd122907bbeb1ef4ec4db"
OOD_BANKING_PROBABILITY_THRESHOLD = 0.5


class DualHeadRouterModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, intent_count: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.domain_head = nn.Linear(hidden_size, 2)
        self.intent_head = nn.Linear(hidden_size, intent_count)

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
        intent_labels: tuple[str, ...],
        threshold: float,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model.to("cpu").eval()
        self.intent_labels = intent_labels
        self.threshold = threshold
        self.max_length = max_length

    @classmethod
    def from_hub(cls) -> LearnedBankingRouter:
        root = Path(snapshot_download(ROUTER_REPO_ID, revision=ROUTER_REVISION))
        config = verify_artifact(root)
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
        labels = tuple(str(label) for label in config["intent_labels"])
        model = DualHeadRouterModel(
            encoder,
            int(encoder.config.hidden_size),
            len(labels),
        )
        heads = load_file(root / "classifier_heads.safetensors", device="cpu")
        model.domain_head.load_state_dict(
            {
                "weight": heads["domain_head.weight"],
                "bias": heads["domain_head.bias"],
            }
        )
        model.intent_head.load_state_dict(
            {
                "weight": heads["intent_head.weight"],
                "bias": heads["intent_head.bias"],
            }
        )
        return cls(
            tokenizer=tokenizer,
            model=model,
            intent_labels=labels,
            threshold=float(config["domain_threshold"]),
            max_length=int(config["max_length"]),
        )

    def classify(
        self,
        message: str,
        history: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        normalized_message = message.strip()
        current = self._predict(f"[CURRENT]\n{normalized_message}")
        current["context_applied"] = False
        if current["route"] == "in_domain":
            return current

        exchanges = _recent_exchanges(history)
        if not exchanges:
            return current
        previous_user, previous_assistant = exchanges[-1]
        if (
            _has_contextual_reference(normalized_message)
            and _is_ambiguous_contextual_fragment(normalized_message)
        ):
            context_reason = "contextual_reference"
        elif (
            _is_short_follow_up(normalized_message)
            and _is_ambiguous_contextual_fragment(normalized_message)
            and _invites_follow_up(previous_assistant)
        ):
            context_reason = "short_follow_up"
        else:
            return current

        previous = self._predict(f"[CURRENT]\n{previous_user}")
        if previous["route"] == "out_of_domain":
            if len(exchanges) < 2:
                return current
            earlier_user, earlier_assistant = exchanges[-2]
            previous_was_follow_up = (
                _has_contextual_reference(previous_user)
                or (
                    _is_short_follow_up(previous_user)
                    and _invites_follow_up(earlier_assistant)
                )
            )
            if not previous_was_follow_up:
                return current
            earlier = self._predict(f"[CURRENT]\n{earlier_user}")
            if earlier["route"] == "out_of_domain":
                return current

        contextual = self._predict(
            f"[CURRENT]\n{normalized_message}\n"
            f"[PREVIOUS_ASSISTANT]\n{previous_assistant}\n"
            f"[PREVIOUS_USER]\n{previous_user}"
        )
        if contextual["route"] == "out_of_domain":
            contextual["route"] = "uncertain"
            contextual["context_route_override"] = (
                "Active banking follow-up was retained for 9B adjudication."
            )
        contextual["context_applied"] = True
        contextual["context_reason"] = context_reason
        contextual["context_chain_depth"] = (
            2 if previous["route"] == "out_of_domain" else 1
        )
        return contextual

    def _predict(self, rendered: str) -> dict[str, Any]:
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        with torch.inference_mode():
            domain_logits, intent_logits = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        banking_probability = float(torch.softmax(domain_logits, dim=-1)[0, 1])
        intent_probabilities = torch.softmax(intent_logits, dim=-1)[0]
        candidate_count = min(3, len(self.intent_labels))
        candidate_probabilities, candidate_indices = torch.topk(
            intent_probabilities,
            k=candidate_count,
        )
        candidates = [
            {
                "intent": self.intent_labels[int(index)],
                "probability": float(probability),
            }
            for probability, index in zip(
                candidate_probabilities,
                candidate_indices,
                strict=True,
            )
        ]
        ood_probability = 1.0 - banking_probability
        ood_threshold = min(self.threshold, OOD_BANKING_PROBABILITY_THRESHOLD)
        in_domain_threshold = max(
            self.threshold,
            OOD_BANKING_PROBABILITY_THRESHOLD,
        )
        if banking_probability >= in_domain_threshold:
            route = "in_domain"
        elif banking_probability < ood_threshold:
            route = "out_of_domain"
        else:
            route = "uncertain"
        return {
            "route": route,
            "banking_probability": banking_probability,
            "ood_probability": ood_probability,
            "confidence": max(banking_probability, ood_probability),
            "intent": candidates[0]["intent"],
            "intent_confidence": candidates[0]["probability"],
            "intent_candidates": candidates,
            "threshold": self.threshold,
            "ood_threshold": ood_threshold,
            "in_domain_threshold": in_domain_threshold,
            "router_revision": ROUTER_REVISION,
        }


def verify_artifact(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("contract") != "banking-dual-head-router-artifact"
        or manifest.get("release_eligible") is not True
    ):
        raise ValueError("router artifact is not release eligible")
    for entry in manifest["files"]:
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe router artifact path")
        path = root / relative
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"router artifact file mismatch: {relative}")
        if _sha256(path) != str(entry["sha256"]):
            raise ValueError(f"router artifact digest mismatch: {relative}")
    config = json.loads((root / "router_config.json").read_text(encoding="utf-8"))
    if config.get("contract") != "banking-dual-head-router":
        raise ValueError("unexpected router configuration")
    return config


def _has_contextual_reference(text: str) -> bool:
    normalized = "".join(character.lower() if character.isalnum() else " " for character in text)
    return bool(
        set(normalized.split())
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


def _recent_exchanges(
    history: list[dict[str, Any]] | None,
) -> list[tuple[str, str]]:
    messages = [
        (str(item.get("role")), str(item.get("content")).strip())
        for item in (history or [])
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and str(item["content"]).strip()
    ]
    exchanges: list[tuple[str, str]] = []
    active_user: str | None = None
    active_assistant: str | None = None
    for role, content in messages:
        if role == "user":
            if active_user is not None and active_assistant is not None:
                exchanges.append((active_user, active_assistant))
            active_user = content
            active_assistant = None
        elif active_user is not None:
            active_assistant = content
    if active_user is not None and active_assistant is not None:
        exchanges.append((active_user, active_assistant))
    return exchanges


def _is_short_follow_up(text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", text)
    return 0 < len(words) <= 6 and len(text) <= 64


def _is_ambiguous_contextual_fragment(text: str) -> bool:
    words = {word.casefold() for word in re.findall(r"[A-Za-z0-9]+", text)}
    if not words:
        return False
    fragment_vocabulary = {
        "a",
        "about",
        "again",
        "also",
        "an",
        "and",
        "another",
        "are",
        "be",
        "block",
        "can",
        "cancel",
        "care",
        "check",
        "could",
        "did",
        "dispute",
        "do",
        "does",
        "else",
        "for",
        "freeze",
        "happen",
        "happened",
        "is",
        "it",
        "list",
        "make",
        "me",
        "my",
        "need",
        "next",
        "no",
        "now",
        "of",
        "ok",
        "okay",
        "one",
        "ones",
        "or",
        "other",
        "please",
        "replace",
        "same",
        "show",
        "stop",
        "sure",
        "take",
        "tell",
        "that",
        "the",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "want",
        "was",
        "were",
        "what",
        "which",
        "will",
        "would",
        "yes",
    }
    return all(word.isdigit() or word in fragment_vocabulary for word in words)


def _invites_follow_up(text: str) -> bool:
    normalized = text.casefold()
    return "?" in text or any(
        phrase in normalized
        for phrase in (
            "please provide",
            "please tell me",
            "which one",
            "what are",
            "could you provide",
            "can you provide",
            "need the",
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
