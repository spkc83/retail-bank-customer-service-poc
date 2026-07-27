from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_router() -> ModuleType:
    path = Path("spaces/hello-banking-moe-9b/router.py")
    spec = importlib.util.spec_from_file_location("hello_banking_space_router", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_messages_for_route_preserves_history_and_appends_current_turn() -> None:
    router = load_router()
    history = [
        {"role": "user", "content": "My transfer is pending."},
        {"role": "assistant", "content": "Check its status."},
        {"role": "system", "content": "ignored"},
    ]

    messages = router.messages_for_route("What should I do next?", history)

    assert [(message.role, message.content) for message in messages] == [
        ("user", "My transfer is pending."),
        ("assistant", "Check its status."),
        ("user", "What should I do next?"),
    ]


def test_messages_for_route_rejects_malformed_public_api_payloads() -> None:
    router = load_router()

    with pytest.raises(ValueError, match="non-empty string"):
        router.messages_for_route(None, [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="history must be a list"):
        router.messages_for_route("Where is my card?", {"role": "user"})  # type: ignore[arg-type]
