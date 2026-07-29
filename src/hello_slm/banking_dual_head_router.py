from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn

from hello_slm.banking_policy import ChatMessage, DomainRouteResult

ROUTER_REPO_ID = "spkc83/retail-bank-domain-intent-router"
ROUTER_REVISION = "136ee159d19cda7f585dd122907bbeb1ef4ec4db"


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
    """Manifest-verified banking domain and Banking77 intent classifier."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: DualHeadRouterModel,
        intent_labels: Sequence[str],
        domain_threshold: float,
        max_length: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if not intent_labels:
            raise ValueError("intent_labels must not be empty")
        if not 0.0 < domain_threshold < 1.0:
            raise ValueError("domain_threshold must be between zero and one")
        self.tokenizer = tokenizer
        self.model = model
        self.intent_labels = tuple(intent_labels)
        self.domain_threshold = domain_threshold
        self.max_length = max_length
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> LearnedBankingRouter:
        from transformers import AutoModel, AutoTokenizer

        root = Path(artifact_dir)
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
            device=device,
        )

    @classmethod
    def from_hub(
        cls,
        *,
        repo_id: str = ROUTER_REPO_ID,
        revision: str = ROUTER_REVISION,
        device: torch.device | str = "cpu",
        token: str | None = None,
    ) -> LearnedBankingRouter:
        from huggingface_hub import snapshot_download

        artifact_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
        )
        return cls.from_artifact_dir(artifact_dir, device=device)

    def classify(self, messages: Sequence[ChatMessage]) -> DomainRouteResult:
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

    def _predict(self, rendered: str) -> DomainRouteResult:
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {
            name: tensor.to(self.device)
            for name, tensor in encoded.items()
            if name in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode():
            domain_logits, intent_logits = self.model(**inputs)
        banking_probability = float(
            torch.softmax(domain_logits.float(), dim=-1)[0, 1].cpu()
        )
        accepted = banking_probability >= self.domain_threshold
        intent_probabilities = torch.softmax(intent_logits.float(), dim=-1)[0]
        intent_confidence_tensor, intent_index_tensor = intent_probabilities.max(dim=-1)
        intent_index = int(intent_index_tensor.cpu())
        intent_confidence = float(intent_confidence_tensor.cpu())
        return DomainRouteResult(
            route="in_domain" if accepted else "out_of_domain",
            confidence=banking_probability if accepted else 1.0 - banking_probability,
            intent=self.intent_labels[intent_index] if accepted else None,
            reason=(
                f"learned domain probability {banking_probability:.6f} "
                f"{'>=' if accepted else '<'} threshold {self.domain_threshold:.6f}"
            ),
            banking_probability=banking_probability,
            intent_confidence=intent_confidence,
        )


def render_conversation_input(messages: Sequence[ChatMessage]) -> str:
    user_texts = [message.content.strip() for message in messages if message.role == "user"]
    if not user_texts or not user_texts[-1]:
        raise ValueError("at least one non-empty user message is required")
    rendered = f"[CURRENT]\n{user_texts[-1]}"
    if len(user_texts) > 1 and user_texts[-2]:
        rendered += f"\n[PREVIOUS_USER]\n{user_texts[-2]}"
    return rendered


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


def verify_router_artifact(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    config_path = root / "router_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("contract") != "banking-dual-head-router":
        raise ValueError("unexpected router configuration contract")
    if config.get("fail_closed") is not True:
        raise ValueError("router artifact does not require fail-closed serving")
    return config


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
