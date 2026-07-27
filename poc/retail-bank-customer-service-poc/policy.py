from __future__ import annotations

import re

OOD_RESPONSE = (
    "I can only help with this synthetic retail-banking demo. "
    "Try asking about the signed-in demo customer’s accounts, cards, transactions, "
    "transfers, or service cases."
)
SENSITIVE_RESPONSE = (
    "Do not enter real or synthetic PINs, CVV/CVC values, passwords, one-time codes, "
    "or complete card/account numbers. This demo never needs those credentials."
)
MODEL_FAILURE_RESPONSE = (
    "The model could not produce a safe, valid banking tool call. No synthetic account "
    "data was changed. Please try one of the preset requests."
)

SECRET_VALUE = re.compile(
    r"\b(?:my\s+)?(?:pin|cvv|cvc|password|passcode|otp|one[- ]time code)\s*"
    r"(?:is|=|:)\s*[A-Za-z0-9-]{3,}\b",
    re.IGNORECASE,
)
LONG_NUMBER = re.compile(r"\b(?:\d[ -]?){12,19}\b")
UNSAFE_OUTPUT = re.compile(
    r"\b(?:provide|send|share|tell (?:me|us)|enter)\b.{0,80}"
    r"\b(?:account number|card number|cvv|cvc|pin|password|passcode|otp)\b",
    re.IGNORECASE | re.DOTALL,
)


def contains_sensitive_value(message: str) -> bool:
    return bool(SECRET_VALUE.search(message) or LONG_NUMBER.search(message))


def generated_response_is_unsafe(response: str) -> bool:
    return bool(UNSAFE_OUTPUT.search(response))
