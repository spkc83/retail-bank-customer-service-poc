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
