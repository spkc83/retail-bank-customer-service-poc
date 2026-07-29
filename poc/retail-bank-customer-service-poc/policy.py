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
    "The 9B conversational model is currently unavailable, so this turn could not "
    "be completed. No CPU-generated banking answer was substituted."
)

SECRET_VALUE = re.compile(
    r"\b(?:my\s+)?(?:pin|cvv|cvc|password|passcode|otp|one[- ]time code)\s*"
    r"(?:is|=|:)\s*[A-Za-z0-9-]{3,}\b",
    re.IGNORECASE,
)
LONG_NUMBER = re.compile(r"\b(?:\d[ -]?){12,19}\b")


def contains_sensitive_value(message: str) -> bool:
    return bool(SECRET_VALUE.search(message) or LONG_NUMBER.search(message))
