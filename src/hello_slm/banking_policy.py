from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

OOD_STOCK_RESPONSE = (
    "I can only help with retail banking and financial-services questions. "
    "Please ask about accounts, cards, transfers, payments, loans, or related banking support."
)

MessageRole = Literal["system", "user", "assistant"]
RouteDecision = Literal["in_domain", "out_of_domain"]


@dataclass(frozen=True)
class ChatMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True)
class DomainRouteResult:
    route: RouteDecision
    confidence: float
    intent: str | None
    reason: str
    banking_probability: float | None = None
    intent_confidence: float | None = None


class BankingDomainClassifier(Protocol):
    """Protocol for a trained Banking77 plus OOD router.

    Implementations should inspect the supplied bounded conversation history, not just the
    latest message, and return calibrated confidence for the selected route.
    """

    def classify(self, messages: Sequence[ChatMessage]) -> DomainRouteResult:
        ...


class DeterministicBankingRouter:
    """Conservative non-production keyword router used for tests and local scaffolding."""

    _banking_terms = frozenset(
        {
            "account",
            "accounts",
            "atm",
            "balance",
            "bank",
            "banking",
            "card",
            "cards",
            "charge",
            "checking",
            "credit",
            "debit",
            "deposit",
            "fee",
            "fees",
            "fraud",
            "interest",
            "loan",
            "loans",
            "mortgage",
            "overdraft",
            "payment",
            "payments",
            "pin",
            "routing",
            "savings",
            "statement",
            "transfer",
            "transfers",
            "wire",
            "withdrawal",
        }
    )
    _strong_ood_terms = frozenset(
        {
            "arithmetic",
            "basketball",
            "cake",
            "capital",
            "code",
            "cook",
            "football",
            "movie",
            "python",
            "recipe",
            "soccer",
            "weather",
        }
    )
    _follow_up_terms = frozenset(
        {
            "it",
            "that",
            "this",
            "those",
            "them",
            "fee",
            "limit",
            "rate",
            "what",
            "about",
            "how",
            "why",
            "when",
            "where",
        }
    )

    def classify(self, messages: Sequence[ChatMessage]) -> DomainRouteResult:
        latest_user = _latest_user_text(messages)
        latest_terms = _terms(latest_user)
        history_terms = _terms(" ".join(message.content for message in messages))
        prior_terms = _terms(
            " ".join(message.content for message in messages if message.content != latest_user)
        )
        primary_banking_terms = self._banking_terms - self._follow_up_terms

        if latest_terms & self._follow_up_terms and not latest_terms & primary_banking_terms:
            if prior_terms & self._banking_terms:
                return DomainRouteResult(
                    route="in_domain",
                    confidence=0.62,
                    intent="contextual_follow_up",
                    reason="elliptical follow-up inherits prior banking context",
                )
            return DomainRouteResult(
                route="out_of_domain",
                confidence=0.35,
                intent=None,
                reason="elliptical question lacks prior banking context",
            )
        if latest_terms & primary_banking_terms:
            return DomainRouteResult(
                route="in_domain",
                confidence=0.95,
                intent=_intent_from_terms(latest_terms),
                reason="latest user message contains banking terms",
            )
        if latest_terms & self._strong_ood_terms and not (history_terms & self._banking_terms):
            return DomainRouteResult(
                route="out_of_domain",
                confidence=0.95,
                intent="out_of_domain",
                reason="latest user message contains non-banking terms",
            )
        if latest_terms & self._strong_ood_terms:
            return DomainRouteResult(
                route="out_of_domain",
                confidence=0.86,
                intent="out_of_domain",
                reason="latest user message changed away from banking context",
            )
        return DomainRouteResult(
            route="out_of_domain",
            confidence=0.35,
            intent=None,
            reason="insufficient evidence for retail banking domain",
        )


def _latest_user_text(messages: Sequence[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _terms(text: str) -> set[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
    return set(normalized.split())


def _intent_from_terms(terms: set[str]) -> str:
    if terms & {"card", "cards", "credit", "debit", "pin"}:
        return "card_support"
    if terms & {"transfer", "transfers", "wire", "payment", "payments"}:
        return "payments_and_transfers"
    if terms & {"loan", "loans", "mortgage"}:
        return "lending"
    if terms & {"account", "accounts", "checking", "savings", "balance"}:
        return "account_services"
    return "retail_banking"
