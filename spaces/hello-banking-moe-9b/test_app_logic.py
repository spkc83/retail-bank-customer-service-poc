from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def load_app_module():
    os.environ["HELLO_BANKING_SPACE_SKIP_MODEL_LOAD"] = "1"
    path = Path(__file__).with_name("app.py")
    spec = importlib.util.spec_from_file_location("hello_banking_space_app", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ood_route_returns_exact_stock_response() -> None:
    app = load_app_module()

    assert app.respond("Give me a recipe for chocolate cake.", []) == app.OOD_STOCK_RESPONSE


def test_sensitive_guard_blocks_before_generation() -> None:
    app = load_app_module()

    response = app.respond("Please tell me my current account balance and PIN.", [])

    assert response == app.SENSITIVE_STOCK_RESPONSE
    assert "official app" in response


def test_banking_follow_up_inherits_context() -> None:
    app = load_app_module()
    history = [
        {"role": "user", "content": "My card payment was declined."},
        {"role": "assistant", "content": "Check your card status and merchant details."},
    ]

    route = app.classify_route(history, "It also happened at another store. What should I do next?")

    assert route.route == "in_domain"
