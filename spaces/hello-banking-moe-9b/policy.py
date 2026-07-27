"""Deterministic safety boundary for the public banking model demo."""

from __future__ import annotations

import re

OOD_RESPONSE = (
    "I can only help with retail banking and financial-services questions. "
    "Please ask about accounts, cards, transfers, payments, loans, or related banking support."
)
SENSITIVE_RESPONSE = (
    "I can’t access your account or accept sensitive credentials. Never share your PIN, "
    "CVV/CVC, password, one-time code, or full card or account number here. Use your bank’s "
    "official app or website, or call the verified number on the back of your card."
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


def is_sensitive(message: str) -> bool:
    return bool(SENSITIVE_TERMS.search(message) or LONG_NUMBER.search(message))


def generated_response_is_unsafe(response: str) -> bool:
    return bool(UNSAFE_OUTPUT.search(response))
