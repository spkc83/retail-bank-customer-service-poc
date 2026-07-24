from __future__ import annotations

from hello_slm.banking_policy import (
    OOD_STOCK_RESPONSE,
    ChatMessage,
    DeterministicBankingRouter,
)


def test_exact_ood_stock_response_contract() -> None:
    assert OOD_STOCK_RESPONSE == (
        "I can only help with retail banking and financial-services questions. "
        "Please ask about accounts, cards, transfers, payments, loans, or related banking support."
    )


def test_router_accepts_current_banking_message() -> None:
    route = DeterministicBankingRouter().classify(
        [ChatMessage(role="user", content="How do I replace my debit card?")]
    )

    assert route.route == "in_domain"
    assert route.confidence >= 0.8
    assert route.intent == "card_support"


def test_router_uses_full_history_for_elliptical_follow_up() -> None:
    route = DeterministicBankingRouter().classify(
        [
            ChatMessage(role="user", content="I need help with a debit card replacement."),
            ChatMessage(role="assistant", content="I can help with card support."),
            ChatMessage(role="user", content="What about the fee?"),
        ]
    )

    assert route.route == "in_domain"
    assert 0.5 <= route.confidence < 0.8
    assert route.intent == "contextual_follow_up"


def test_router_rejects_low_confidence_non_banking_message() -> None:
    route = DeterministicBankingRouter().classify(
        [ChatMessage(role="user", content="Tell me something nice.")]
    )

    assert route.route == "out_of_domain"
    assert route.confidence < 0.5
