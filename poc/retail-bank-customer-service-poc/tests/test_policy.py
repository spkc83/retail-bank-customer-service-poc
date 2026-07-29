from __future__ import annotations

from policy import contains_sensitive_value


def test_sensitive_guard_blocks_values_but_allows_support_questions() -> None:
    assert contains_sensitive_value("My PIN is 1234")
    assert contains_sensitive_value("Use card 4111 1111 1111 1111")
    assert not contains_sensitive_value("I forgot my PIN. What should I do?")
    assert not contains_sensitive_value("Show my synthetic account balances.")
