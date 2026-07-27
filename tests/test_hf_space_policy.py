from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_policy() -> ModuleType:
    path = Path("spaces/hello-banking-moe-9b/policy.py")
    spec = importlib.util.spec_from_file_location("hello_banking_space_policy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_domain_route_handles_banking_ood_and_contextual_follow_up() -> None:
    policy = load_policy()
    history = [
        {"role": "user", "content": "My card was declined."},
        {"role": "assistant", "content": "Contact your bank."},
    ]

    assert policy.is_in_domain("How do I replace a debit card?", []) is True
    assert policy.is_in_domain("What should I do next?", history) is True
    assert policy.is_in_domain("What is the weather?", history) is False
    assert policy.is_in_domain("Write Python code.", []) is False


def test_sensitive_input_and_unsafe_generated_requests_are_blocked() -> None:
    policy = load_policy()

    assert policy.is_sensitive("My PIN is 1234") is True
    assert policy.is_sensitive("My card is 4111 1111 1111 1111") is True
    assert policy.is_sensitive("How do I replace my card?") is False
    assert policy.generated_response_is_unsafe(
        "Please provide your card number and CVV so I can help."
    )
    assert not policy.generated_response_is_unsafe(
        "Use the verified number on the back of your card."
    )
