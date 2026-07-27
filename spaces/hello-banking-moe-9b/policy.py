"""Deterministic safety boundary for the public banking model demo."""

from __future__ import annotations

import re
from typing import Any

OOD_RESPONSE = (
    "I can only help with retail banking and financial-services questions. "
    "Please ask about accounts, cards, transfers, payments, loans, or related banking support."
)
SENSITIVE_RESPONSE = (
    "I can’t access your account or accept sensitive credentials. Never share your PIN, "
    "CVV/CVC, password, one-time code, or full card or account number here. Use your bank’s "
    "official app or website, or call the verified number on the back of your card."
)

BANKING_TERMS = frozenset(
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
STRONG_OOD_TERMS = frozenset(
    {
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
FOLLOW_UP_TERMS = frozenset(
    {
        "about",
        "again",
        "how",
        "it",
        "that",
        "them",
        "this",
        "those",
        "what",
        "when",
        "where",
        "why",
    }
)
SENSITIVE_TERMS = re.compile(
    r"\b(pin|cvv|cvc|password|passcode|one[- ]time (?:code|password)|otp|"
    r"account balance|my balance|account number|routing number|card number)\b",
    re.IGNORECASE,
)
LONG_NUMBER = re.compile(r"\b(?:\d[ -]?){12,19}\b")
UNSAFE_OUTPUT = re.compile(
    r"\b(?:provide|send|share|tell (?:me|us)|enter)\b.{0,80}"
    r"\b(?:account details|account number|card number|cvv|cvc|pin|password|passcode|otp)\b",
    re.IGNORECASE | re.DOTALL,
)


def terms(text: str) -> set[str]:
    normalized = "".join(character.lower() if character.isalnum() else " " for character in text)
    return set(normalized.split())


def user_texts(history: list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("content", ""))
        for item in history
        if item.get("role") == "user" and isinstance(item.get("content"), str)
    ]


def is_sensitive(message: str) -> bool:
    return bool(SENSITIVE_TERMS.search(message) or LONG_NUMBER.search(message))


def is_in_domain(message: str, history: list[dict[str, Any]]) -> bool:
    latest_terms = terms(message)
    if latest_terms & STRONG_OOD_TERMS:
        return False
    if latest_terms & BANKING_TERMS:
        return True
    if latest_terms & FOLLOW_UP_TERMS:
        prior_terms = terms(" ".join(user_texts(history)))
        return bool(prior_terms & BANKING_TERMS)
    return False


def generated_response_is_unsafe(response: str) -> bool:
    return bool(UNSAFE_OUTPUT.search(response))
