from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv(
        "DEMO_AUTH_JSON",
        '{"alex.demo":"alex-test-password","maya.demo":"maya-test-password"}',
    )
    monkeypatch.setenv("POC_SKIP_MODEL_LOAD", "1")
    monkeypatch.setenv("POC_SKIP_ROUTER_LOAD", "1")
    monkeypatch.setenv("POC_SESSION_DB_DIR", str(tmp_path / "sessions"))
    for name in (
        "app",
        "model_service",
        "state",
        "zero_gpu_runtime",
    ):
        sys.modules.pop(name, None)
    return importlib.import_module("app")


def request(username: str = "alex.demo", session_hash: str = "browser-session"):
    return SimpleNamespace(username=username, session_hash=session_hash)


def route(
    route_name: str = "in_domain",
    *,
    banking_probability: float = 0.99,
    intent: str = "pending_transfer",
) -> dict[str, object]:
    return {
        "route": route_name,
        "banking_probability": banking_probability,
        "ood_probability": 1 - banking_probability,
        "confidence": max(banking_probability, 1 - banking_probability),
        "intent": intent,
        "intent_confidence": 0.8,
        "intent_candidates": [
            {"intent": intent, "probability": 0.8},
            {"intent": "cash_withdrawal", "probability": 0.1},
            {"intent": "card_payment_fee_charged", "probability": 0.05},
        ],
        "threshold": 0.98,
        "ood_threshold": 0.02,
        "router_revision": "test-router",
    }


def test_app_constructs_expected_authenticated_api_surface(app_module) -> None:
    api_names = {
        dependency.get("api_name")
        for dependency in app_module.demo.config["dependencies"]
    }

    assert {"chat", "route", "customer_snapshot", "reset_demo"} <= api_names
    assert {username for username, _ in app_module.AUTH_CREDENTIALS} == {
        "alex.demo",
        "maya.demo",
    }


def test_only_registered_model_turn_is_the_zero_gpu_boundary(app_module) -> None:
    assert not hasattr(app_module.dispatch_turn, "_zero_gpu_config")
    assert app_module.finalize_turn._zero_gpu_config == {
        "size": "large",
        "duration": 90,
    }
    assert not hasattr(app_module.generate_text, "_zero_gpu_config")


def test_greeting_and_uncertain_turns_schedule_9b_instead_of_cpu_response(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: route("uncertain", banking_probability=0.52, intent="small_talk"),
    )

    result = app_module.dispatch_turn("yo, sup ?", [], [], 4, request())

    assert result[2] == []
    assert result[1][-1]["role"] == "assistant"
    assert "9B model" in result[1][-1]["content"]
    pending = result[6]
    assert pending["message"] == "yo, sup ?"
    assert pending["conversation"] == []
    assert pending["router_result"]["intent"] == "small_talk"
    assert pending["turn_id"]
    assert pending["epoch"] == 4
    assert "username" not in pending
    assert "session_hash" not in pending


def test_high_confidence_ood_uses_stock_response_without_gpu(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: route(
            "out_of_domain",
            banking_probability=0.001,
            intent="cash_withdrawal",
        ),
    )

    result = app_module.dispatch_turn("Write a Python web scraper.", [], [], 0, request())

    assert "synthetic retail-banking" in result[1][-1]["content"]
    assert result[1] == result[2]
    assert result[6] == pytest.importorskip("gradio").skip()
    assert "out_of_domain" in result[5]


def test_router_unavailability_fails_open_to_9b_experiment(app_module) -> None:
    result = app_module.route_query("hello", [])

    assert result["route"] == "uncertain"
    assert result["intent"] is None
    assert result["intent_candidates"] == []


def test_registered_gpu_turn_returns_model_authored_greeting(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 50)
    monkeypatch.setattr(
        app_module,
        "generate_text",
        lambda *_args: "Hey! How can I help with your banking today?",
    )
    pending = {
        "turn_id": "turn-1",
        "message": "yo, sup?",
        "conversation": [],
        "router_result": route("uncertain", banking_probability=0.5, intent="small_talk"),
        "epoch": 2,
    }

    result = app_module.finalize_turn(pending, 2, [], [], request())

    assert result[0] == result[1]
    assert result[0][-1]["content"] == "Hey! How can I help with your banking today?"
    assert "model authored" in result[3]
    assert "small_talk" in result[4]


def test_model_selects_transfer_tool_and_receives_full_tool_history(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        [
            '<tool_call>{"name": "list_transfers", "arguments": {}}</tool_call>',
            (
                "River Consulting has a pending USD 450 transfer, and Jamie Lee "
                "has a completed USD 125 transfer."
            ),
        ]
    )
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(app_module, "generate_text", lambda *_args: next(outputs))
    pending = {
        "turn_id": "turn-2",
        "message": "What transfers are there on my account?",
        "conversation": [],
        "router_result": route(),
        "epoch": 3,
    }

    result = app_module.finalize_turn(pending, 3, [], [], request())

    canonical = result[1]
    assert [item["role"] for item in canonical] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert canonical[1]["tool_calls"][0]["function"]["name"] == "list_transfers"
    assert "River Consulting" in result[0][-1]["content"]
    assert "list_transfers" in result[4]


def test_gpu_failure_never_generates_cpu_servicing_answer(
    app_module,
) -> None:
    pending = {
        "turn_id": "failed-read",
        "message": "What transfers are there on my account?",
        "conversation": [],
        "router_result": route(),
        "epoch": 5,
    }

    result = app_module.fail_pending_turn(pending, 5, [], [], request())

    response = result[0][-1]["content"]
    assert response == app_module.MODEL_FAILURE_RESPONSE
    assert "River Consulting" not in response
    assert "No CPU-generated banking answer was substituted" in response
    assert result[0] == result[1]
    assert [item["role"] for item in result[1]] == ["user", "assistant"]
    assert "could not allocate" in result[3].lower()


def test_second_pass_failure_preserves_executed_write_in_history_and_diagnostics(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        [
            '<tool_call>{"name": "freeze_card", "arguments": {"last4": "4821"}}</tool_call>',
            "",
        ]
    )
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(app_module, "generate_text", lambda *_args: next(outputs))
    pending = {
        "turn_id": "failed-second-pass",
        "message": "Freeze card 4821.",
        "conversation": [],
        "router_result": route(intent="cash_withdrawal"),
        "epoch": 5,
    }

    result = app_module.finalize_turn(pending, 5, [], [], request())

    assert [item["role"] for item in result[1]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert "freeze_card" in result[4]
    assert "success" in result[4]
    assert "`frozen`" in result[2]
    assert result[0][-1]["content"] == app_module.MODEL_FAILURE_RESPONSE


def test_stale_gpu_turn_after_reset_executes_nothing(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def should_not_run(*_args, **_kwargs):
        raise AssertionError("stale turn must not run the model or a tool")

    monkeypatch.setattr(app_module, "generate_text", should_not_run)
    visible = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Fresh reset session"},
    ]
    pending = {
        "turn_id": "old-turn",
        "message": "Cancel my pending transfer.",
        "conversation": [],
        "router_result": route(),
        "epoch": 3,
    }

    result = app_module.finalize_turn(pending, 4, visible, visible, request())

    assert result[0] == visible
    assert result[1] == visible
    assert "expired" in result[3].lower()


def test_sensitive_value_is_rejected_before_router_or_model(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def should_not_run(*_args, **_kwargs):
        raise AssertionError("router must not run")

    monkeypatch.setattr(app_module, "route_query", should_not_run)

    result = app_module.dispatch_turn("My PIN is 1234", [], [], 0, request())

    assert "never needs" in result[1][-1]["content"]
    assert result[1] == result[2]
    assert result[6] == pytest.importorskip("gradio").skip()


def test_reset_clears_visible_and_canonical_history_and_advances_epoch(
    app_module,
) -> None:
    result = app_module.reset_session(7, request())

    assert result[0] == []
    assert result[1] == []
    assert result[4] == 8
