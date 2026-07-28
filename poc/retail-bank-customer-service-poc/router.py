from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROUTER_REPO_ID = "spkc83/retail-bank-domain-intent-router"
ROUTER_REVISION = "e7d928e5cf8c8be0883625f276c4e6c85c35eaf1"


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
        prior_users = [
            str(item["content"]).strip()
            for item in (history or [])
            if isinstance(item, dict)
            and item.get("role") == "user"
            and isinstance(item.get("content"), str)
        ]
        current = self._predict(f"[CURRENT]\n{message.strip()}")
        if current["route"] == "in_domain":
            return current
        if not prior_users or not _has_contextual_reference(message):
            return current
        return self._predict(
            f"[CURRENT]\n{message.strip()}\n[PREVIOUS_USER]\n{prior_users[-1]}"
        )

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
        ood_threshold = 1.0 - self.threshold
        if banking_probability >= self.threshold:
            route = "in_domain"
        elif banking_probability <= ood_threshold:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
