from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hello_slm.banking_dual_head_router import (
    ChatMessage,
    DualHeadRouterModel,
    LearnedBankingRouter,
    render_conversation_input,
    verify_router_artifact,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.rendered = ""

    def __call__(self, text: str, **_kwargs: object) -> dict[str, torch.Tensor]:
        self.rendered = text
        current = text.split("[PREVIOUS_USER]", maxsplit=1)[0]
        is_contextual_followup = "next" in current and "recipient" in text
        token = 1 if "recipient" in current or is_contextual_followup else 0
        return {
            "input_ids": torch.tensor([[token]]),
            "attention_mask": torch.tensor([[1]]),
        }


class FakeEncoder(nn.Module):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        value = input_ids.float().unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=torch.cat((value, 1.0 - value), dim=-1))


def make_router() -> tuple[LearnedBankingRouter, RecordingTokenizer]:
    tokenizer = RecordingTokenizer()
    model = DualHeadRouterModel(FakeEncoder(), hidden_size=2, num_intents=2)
    with torch.no_grad():
        model.domain_head.weight.copy_(torch.tensor([[-4.0, 4.0], [4.0, -4.0]]))
        model.domain_head.bias.zero_()
        model.intent_head.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        model.intent_head.bias.zero_()
    return (
        LearnedBankingRouter(
            tokenizer=tokenizer,
            model=model,
            intent_labels=("card_arrival", "pending_transfer"),
            domain_threshold=0.9,
            max_length=96,
        ),
        tokenizer,
    )


def test_classifier_returns_probability_route_and_intent() -> None:
    router, _ = make_router()

    result = router.classify(
        [ChatMessage(role="user", content="Why has the recipient not received the transfer?")]
    )

    assert result.route == "in_domain"
    assert result.intent == "pending_transfer"
    assert result.banking_probability is not None
    assert result.banking_probability > 0.99


def test_classifier_rejects_ood_and_current_turn_has_priority() -> None:
    router, tokenizer = make_router()
    messages = [
        ChatMessage(role="user", content="Why has the recipient not received the transfer?"),
        ChatMessage(role="assistant", content="Check its status."),
        ChatMessage(role="user", content="Write a poem about penguins."),
    ]

    result = router.classify(messages)

    assert result.route == "out_of_domain"
    assert result.intent is None
    assert tokenizer.rendered == (
        "[CURRENT]\nWrite a poem about penguins."
    )


def test_classifier_uses_previous_user_turn_for_contextual_followup() -> None:
    router, tokenizer = make_router()
    messages = [
        ChatMessage(role="user", content="Why has the recipient not received the transfer?"),
        ChatMessage(role="assistant", content="Check its status."),
        ChatMessage(role="user", content="What should I do next?"),
    ]

    result = router.classify(messages)

    assert result.route == "in_domain"
    assert result.intent == "pending_transfer"
    assert tokenizer.rendered.endswith(
        "[PREVIOUS_USER]\nWhy has the recipient not received the transfer?"
    )


def test_render_input_uses_only_two_most_recent_user_turns() -> None:
    messages = [
        ChatMessage(role="user", content="old"),
        ChatMessage(role="assistant", content="reply"),
        ChatMessage(role="user", content="previous"),
        ChatMessage(role="user", content="current"),
    ]

    assert render_conversation_input(messages) == (
        "[CURRENT]\ncurrent\n[PREVIOUS_USER]\nprevious"
    )


def test_manifest_verification_rejects_corrupt_artifact(tmp_path: Path) -> None:
    config = {
        "contract": "banking-dual-head-router",
        "fail_closed": True,
        "intent_labels": ["card_arrival"],
        "domain_threshold": 0.9,
        "max_length": 96,
    }
    config_path = tmp_path / "router_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest = {
        "contract": "banking-dual-head-router-artifact",
        "release_eligible": True,
        "files": [
            {
                "path": "router_config.json",
                "bytes": config_path.stat().st_size,
                "sha256": digest,
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_router_artifact(tmp_path) == config
    config_path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|digest mismatch"):
        verify_router_artifact(tmp_path)
