from __future__ import annotations

import pytest
import torch

from router import LearnedBankingRouter


class FakeTokenizer:
    def __call__(self, *_args, **_kwargs):
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }


class FakeModel:
    def __init__(self, domain_logits, intent_logits) -> None:
        self.domain_logits = torch.tensor([domain_logits], dtype=torch.float32)
        self.intent_logits = torch.tensor([intent_logits], dtype=torch.float32)

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, **_kwargs):
        return self.domain_logits, self.intent_logits


@pytest.mark.parametrize(
    ("domain_logits", "expected_route"),
    [
        ((-8.0, 8.0), "in_domain"),
        ((0.0, 0.0), "uncertain"),
        ((8.0, -8.0), "out_of_domain"),
    ],
)
def test_router_uses_three_way_domain_decision_and_always_returns_top_intents(
    domain_logits,
    expected_route,
) -> None:
    router = LearnedBankingRouter(
        tokenizer=FakeTokenizer(),
        model=FakeModel(domain_logits, [0.1, 3.0, 1.0, 2.0]),
        intent_labels=("a", "b", "c", "d"),
        threshold=0.98,
        max_length=32,
    )

    result = router.classify("hello", [])

    assert result["route"] == expected_route
    assert result["intent"] == "b"
    assert [item["intent"] for item in result["intent_candidates"]] == ["b", "d", "c"]
    assert result["ood_probability"] == pytest.approx(1 - result["banking_probability"])
    assert result["ood_threshold"] == pytest.approx(0.5)


def test_short_fragment_uses_the_immediately_preceding_banking_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = LearnedBankingRouter(
        tokenizer=FakeTokenizer(),
        model=FakeModel([8.0, -8.0], [3.0, 1.0]),
        intent_labels=("card", "other"),
        threshold=0.98,
        max_length=32,
    )
    rendered_inputs: list[str] = []

    def predict(rendered: str):
        rendered_inputs.append(rendered)
        if rendered == "[CURRENT]\n4821":
            return {"route": "out_of_domain", "intent": "other"}
        if rendered == "[CURRENT]\nMy card was stolen. Freeze it.":
            return {"route": "in_domain", "intent": "card"}
        return {"route": "uncertain", "intent": "card"}

    monkeypatch.setattr(router, "_predict", predict)

    result = router.classify(
        "4821",
        [
            {"role": "user", "content": "My card was stolen. Freeze it."},
            {
                "role": "assistant",
                "content": "What are the last four digits of the card?",
            },
        ],
    )

    assert result["route"] == "uncertain"
    assert result["context_applied"] is True
    assert result["context_reason"] == "short_follow_up"
    assert "[PREVIOUS_ASSISTANT]\nWhat are the last four digits" in rendered_inputs[-1]
    assert "[PREVIOUS_USER]\nMy card was stolen. Freeze it." in rendered_inputs[-1]


def test_short_fragment_does_not_inherit_an_out_of_domain_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = LearnedBankingRouter(
        tokenizer=FakeTokenizer(),
        model=FakeModel([8.0, -8.0], [3.0, 1.0]),
        intent_labels=("card", "other"),
        threshold=0.98,
        max_length=32,
    )
    predictions = iter(
        [
            {"route": "out_of_domain", "intent": "other"},
            {"route": "out_of_domain", "intent": "other"},
        ]
    )
    monkeypatch.setattr(router, "_predict", lambda _rendered: next(predictions))

    result = router.classify(
        "yes",
        [
            {"role": "user", "content": "What is the weather tomorrow?"},
            {
                "role": "assistant",
                "content": "I can only help with the retail-banking demonstration.",
            },
        ],
    )

    assert result["route"] == "out_of_domain"
    assert result.get("context_applied") is not True
