from __future__ import annotations

from policy import contains_sensitive_value, generated_response_is_unsafe


def test_sensitive_guard_blocks_values_but_allows_support_questions() -> None:
    assert contains_sensitive_value("My PIN is 1234")
    assert contains_sensitive_value("Use card 4111 1111 1111 1111")
    assert not contains_sensitive_value("I forgot my PIN. What should I do?")
    assert not contains_sensitive_value("Show my synthetic account balances.")


def test_generated_response_guard_blocks_credential_requests() -> None:
    assert generated_response_is_unsafe("Please provide your card number and CVV.")
    assert not generated_response_is_unsafe(
        "Use the verified number on the back of your synthetic card."
    )
