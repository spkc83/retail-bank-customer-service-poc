from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file
from torch import nn

from hello_slm.banking_conversation_router_data import (
    render_router_input_with_context,
)

MessageRole = Literal["system", "user", "assistant"]
RouteDecision = Literal["in_domain", "out_of_domain", "uncertain"]


@dataclass(frozen=True)
class ChatMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True)
class ConversationRouteResult:
    route: RouteDecision
    confidence: float
    banking_probability: float
    ood_probability: float
    capability: str | None
    capability_confidence: float
    capability_candidates: tuple[dict[str, float | str], ...]
    relation_probabilities: dict[str, float]
    active_relations: tuple[str, ...]
    context_applied: bool
    reason: str


class ConversationRouterModel(nn.Module):
    """One cross-encoder with domain, capability, and multi-label relation heads."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_size: int,
        num_capabilities: int,
        num_relations: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.domain_head = nn.Linear(hidden_size, 2)
        self.capability_head = nn.Linear(hidden_size, num_capabilities)
        self.relation_head = nn.Linear(hidden_size, num_relations)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        return (
            self.domain_head(pooled),
            self.capability_head(pooled),
            self.relation_head(pooled),
        )


class LearnedConversationRouter:
    """History-aware OOD/continuity gate with diagnostic servicing capabilities."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: ConversationRouterModel,
        capability_labels: Sequence[str],
        relation_labels: Sequence[str],
        ood_banking_threshold: float,
        in_domain_threshold: float,
        relation_rescue_threshold: float,
        relation_thresholds: Mapping[str, float] | None = None,
        max_length: int,
        max_exchanges: int = 3,
        device: torch.device | str = "cpu",
    ) -> None:
        if not capability_labels:
            raise ValueError("capability_labels must not be empty")
        if not relation_labels:
            raise ValueError("relation_labels must not be empty")
        if not 0.0 < ood_banking_threshold < in_domain_threshold < 1.0:
            raise ValueError(
                "domain thresholds must satisfy 0 < OOD < in-domain < 1"
            )
        if not 0.0 < relation_rescue_threshold < 1.0:
            raise ValueError("relation_rescue_threshold must be between zero and one")
        if max_length < 32:
            raise ValueError("max_length must be at least 32")
        if max_exchanges < 1:
            raise ValueError("max_exchanges must be positive")

        self.tokenizer = tokenizer
        self.model = model
        self.capability_labels = tuple(capability_labels)
        self.relation_labels = tuple(relation_labels)
        self.ood_banking_threshold = ood_banking_threshold
        self.in_domain_threshold = in_domain_threshold
        self.relation_rescue_threshold = relation_rescue_threshold
        self.relation_thresholds = {
            label: float((relation_thresholds or {}).get(label, 0.5))
            for label in self.relation_labels
        }
        self.max_length = max_length
        self.max_exchanges = max_exchanges
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> LearnedConversationRouter:
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
        capabilities = tuple(str(label) for label in config["capability_labels"])
        relations = tuple(str(label) for label in config["relation_labels"])
        model = ConversationRouterModel(
            encoder,
            hidden_size=int(encoder.config.hidden_size),
            num_capabilities=len(capabilities),
            num_relations=len(relations),
        )
        heads = load_file(root / "classifier_heads.safetensors", device="cpu")
        for name, head in (
            ("domain_head", model.domain_head),
            ("capability_head", model.capability_head),
            ("relation_head", model.relation_head),
        ):
            head.load_state_dict(
                {
                    "weight": heads[f"{name}.weight"],
                    "bias": heads[f"{name}.bias"],
                },
                strict=True,
            )
        return cls(
            tokenizer=tokenizer,
            model=model,
            capability_labels=capabilities,
            relation_labels=relations,
            ood_banking_threshold=float(config["ood_banking_threshold"]),
            in_domain_threshold=float(config["in_domain_threshold"]),
            relation_rescue_threshold=float(config["relation_rescue_threshold"]),
            relation_thresholds=config.get("relation_thresholds"),
            max_length=int(config["max_length"]),
            max_exchanges=int(config.get("max_exchanges", 3)),
            device=device,
        )

    @classmethod
    def from_hub(
        cls,
        *,
        repo_id: str,
        revision: str,
        device: torch.device | str = "cpu",
        token: str | None = None,
    ) -> LearnedConversationRouter:
        from huggingface_hub import snapshot_download

        invalid_character = any(
            character not in "0123456789abcdef" for character in revision
        )
        if len(revision) != 40 or invalid_character:
            raise ValueError("revision must be an immutable 40-character commit")
        root = snapshot_download(repo_id=repo_id, revision=revision, token=token)
        return cls.from_artifact_dir(root, device=device)

    def classify(
        self,
        messages: Sequence[ChatMessage],
    ) -> ConversationRouteResult:
        user_indices = [
            index
            for index, message in enumerate(messages)
            if message.role == "user" and message.content.strip()
        ]
        if not user_indices:
            raise ValueError("at least one non-empty user message is required")
        current_index = user_indices[-1]
        current = messages[current_index].content.strip()
        history = [
            {"role": message.role, "content": message.content.strip()}
            for message in messages[:current_index]
            if message.role in {"user", "assistant"} and message.content.strip()
        ]
        rendered, context_applied = render_router_input_with_context(
            current,
            history,
            max_exchanges=self.max_exchanges,
        )
        return self._predict(rendered, context_applied=context_applied)

    def _predict(
        self,
        rendered: str,
        *,
        context_applied: bool,
    ) -> ConversationRouteResult:
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
            domain_logits, capability_logits, relation_logits = self.model(**inputs)

        banking_probability = float(
            torch.softmax(domain_logits.float(), dim=-1)[0, 1].cpu()
        )
        ood_probability = 1.0 - banking_probability
        relation_values = torch.sigmoid(relation_logits.float())[0].cpu().tolist()
        relations = dict(zip(self.relation_labels, relation_values, strict=True))
        active_relations = tuple(
            label
            for label in self.relation_labels
            if relations[label] >= self.relation_thresholds[label]
        )
        rescue_probability = max(
            relations.get("context_dependent", 0.0),
            relations.get("agent_repair", 0.0),
            relations.get("clarification_answer", 0.0),
        )

        if banking_probability >= self.in_domain_threshold:
            route: RouteDecision = "in_domain"
        elif (
            banking_probability < self.ood_banking_threshold
            and rescue_probability < self.relation_rescue_threshold
        ):
            route = "out_of_domain"
        else:
            route = "uncertain"

        capability_probabilities = torch.softmax(
            capability_logits.float(),
            dim=-1,
        )[0]
        candidate_count = min(3, len(self.capability_labels))
        candidate_probabilities, candidate_indices = torch.topk(
            capability_probabilities,
            k=candidate_count,
        )
        candidate_items: list[dict[str, float | str]] = []
        for probability, index in zip(
            candidate_probabilities.cpu(),
            candidate_indices.cpu(),
            strict=True,
        ):
            candidate_items.append(
                {
                "capability": self.capability_labels[int(index)],
                "probability": float(probability),
                }
            )
        candidates = tuple(candidate_items)
        capability = (
            str(candidates[0]["capability"]) if route == "in_domain" else None
        )
        capability_confidence = float(candidate_probabilities[0].cpu())
        confidence = (
            banking_probability
            if route == "in_domain"
            else ood_probability
            if route == "out_of_domain"
            else max(banking_probability, rescue_probability)
        )
        reason = (
            f"banking={banking_probability:.6f}, "
            f"rescue={rescue_probability:.6f}, "
            f"ood_boundary={self.ood_banking_threshold:.6f}, "
            f"in_domain_boundary={self.in_domain_threshold:.6f}"
        )
        return ConversationRouteResult(
            route=route,
            confidence=confidence,
            banking_probability=banking_probability,
            ood_probability=ood_probability,
            capability=capability,
            capability_confidence=capability_confidence,
            capability_candidates=candidates,
            relation_probabilities=relations,
            active_relations=active_relations,
            context_applied=context_applied,
            reason=reason,
        )


def verify_router_artifact(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    config_path = root / "router_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != "banking-conversation-router-artifact":
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
    if config.get("contract") != "banking-conversation-router":
        raise ValueError("unexpected router configuration contract")
    if int(config.get("format_version", 0)) != 2:
        raise ValueError("unsupported router configuration version")
    return config


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
