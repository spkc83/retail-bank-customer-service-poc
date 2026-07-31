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
    capability: str = "transfers",
) -> dict[str, object]:
    return {
        "route": route_name,
        "banking_probability": banking_probability,
        "ood_probability": 1 - banking_probability,
        "confidence": max(banking_probability, 1 - banking_probability),
        "capability": capability,
        "capability_confidence": 0.8,
        "capability_candidates": [
            {"capability": capability, "probability": 0.8},
            {"capability": "accounts", "probability": 0.1},
            {"capability": "cards", "probability": 0.05},
        ],
        "relation_probabilities": {
            "context_dependent": 0.1,
            "agent_repair": 0.1,
            "topic_shift": 0.1,
            "clarification_answer": 0.1,
        },
        "ood_banking_threshold": 0.2,
        "in_domain_threshold": 0.5,
        "relation_rescue_threshold": 0.5,
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
    assert "alex-test-password" in app_module.AUTH_MESSAGE
    assert "maya-test-password" in app_module.AUTH_MESSAGE


def test_chat_turn_is_a_single_direct_zero_gpu_boundary(app_module) -> None:
    assert app_module.run_model_turn._zero_gpu_config == {
        "size": "large",
        "duration": 90,
    }
    assert not hasattr(app_module.generate_text, "_zero_gpu_config")
    chat_dependencies = [
        dependency
        for dependency in app_module.demo.config["dependencies"]
        if dependency.get("api_name") == "chat"
    ]
    assert len(chat_dependencies) == 1
    assert chat_dependencies[0]["targets"] != []
    assert all(
        dependency.get("trigger_after") is None
        for dependency in app_module.demo.config["dependencies"]
        if dependency.get("api_name") == "chat"
    )


def test_greeting_and_uncertain_turns_are_answered_by_9b(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: route(
            "uncertain",
            banking_probability=0.52,
            capability="conversation",
        ),
    )
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 50)
    monkeypatch.setattr(
        app_module,
        "generate_text",
        lambda *_args: "Hey! How can I help with your banking today?",
    )

    result = app_module.run_model_turn("yo, sup ?", [], [], request())

    assert result[1] == result[2]
    assert result[1][-1]["role"] == "assistant"
    assert result[1][-1]["content"] == "Hey! How can I help with your banking today?"
    assert "model authored" in result[4]
    assert "conversation" in result[5]


def test_high_confidence_ood_uses_stock_response_inside_registered_gpu_turn(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: route(
            "out_of_domain",
            banking_probability=0.001,
            capability="accounts",
        ),
    )

    result = app_module.run_model_turn("Write a Python web scraper.", [], [], request())

    assert "synthetic retail-banking" in result[1][-1]["content"]
    assert result[1] == result[2]
    assert "out_of_domain" in result[5]


def test_test_mode_router_omission_routes_uncertain_to_9b(app_module) -> None:
    result = app_module.route_query("hello", [])

    assert result["route"] == "uncertain"
    assert result["capability"] is None
    assert result["capability_candidates"] == []


def test_classifier_failure_is_visible_and_blocks_9b(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingRouter:
        def classify(self, *_args):
            raise RuntimeError("classifier unavailable")

    monkeypatch.setattr(app_module, "router", FailingRouter())

    def unexpected_generation(*_args):
        pytest.fail("9B generation must not run after classifier failure")

    monkeypatch.setattr(app_module, "generate_text", unexpected_generation)

    result = app_module.run_model_turn(
        "Show my balances.",
        [],
        [],
        request(),
    )

    assert result[1][-1]["content"] == app_module.MODEL_FAILURE_RESPONSE
    assert "classifier failed" in result[4]
    assert "classifier_error" in result[5]
    assert "RuntimeError" in result[5]
    assert "9B generator was not invoked" in result[5]


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
    monkeypatch.setattr(app_module, "route_query", lambda *_args: route())

    result = app_module.run_model_turn(
        "What transfers are there on my account?",
        [],
        [],
        request(),
    )

    canonical = result[2]
    assert [item["role"] for item in canonical] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert canonical[1]["tool_calls"][0]["function"]["name"] == "list_transfers"
    assert "River Consulting" in result[1][-1]["content"]
    assert "list_transfers" in result[5]


def test_direct_answer_is_labeled_with_per_pass_provenance(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(
        app_module,
        "generate_text",
        lambda *_args: "I can explain how savings interest works.",
    )
    monkeypatch.setattr(app_module, "route_query", lambda *_args: route())

    result = app_module.run_model_turn(
        "How does savings interest work?",
        [],
        [],
        request(),
    )

    assert result[1][-1]["content"] == "I can explain how savings interest works."
    assert "9B direct_answer" in result[5]
    assert "`base`" in result[5]
    assert "`reflection`" not in result[5]
    assert "Generation calls: `1`" in result[5]
    assert "prompt SHA-256" in result[5]
    assert "raw output SHA-256" in result[5]
    assert "I can explain how savings interest works." in result[5]
    assert "Runtime device:" in result[5]
    assert "CUDA device name:" in result[5]
    assert app_module.MODEL_ID in result[5]
    assert app_module.MODEL_REVISION in result[5]


def test_gpu_failure_never_generates_cpu_servicing_answer(
    app_module,
) -> None:
    result = app_module.fail_model_turn(
        "What transfers are there on my account?",
        [],
        request(),
    )

    response = result[1][-1]["content"]
    assert response == app_module.MODEL_FAILURE_RESPONSE
    assert "River Consulting" not in response
    assert "No CPU-generated banking answer was substituted" in response
    assert result[1] == result[2]
    assert [item["role"] for item in result[2]] == ["user", "assistant"]
    assert "could not allocate" in result[4].lower()


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
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: route(capability="cards"),
    )

    result = app_module.run_model_turn("Freeze card 4821.", [], [], request())

    assert [item["role"] for item in result[2]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert "freeze_card" in result[5]
    assert "success" in result[5]
    assert "`frozen`" in result[3]
    assert result[1][-1]["content"] == app_module.MODEL_FAILURE_RESPONSE


def test_credential_like_text_reaches_router_and_model(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routed: list[str] = []
    generated: list[str] = []

    def record_route(message, _history):
        routed.append(message)
        return route()

    def record_generation(messages, *_args):
        generated.append(messages[-1]["content"])
        return "I can discuss banking support without using that credential."

    monkeypatch.setattr(
        app_module,
        "route_query",
        record_route,
    )
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(
        app_module,
        "generate_text",
        record_generation,
    )

    result = app_module.run_model_turn("My PIN is 1234", [], [], request())

    assert routed == ["My PIN is 1234"]
    assert generated == ["My PIN is 1234"]
    assert "9B direct_answer" in result[5]
    assert result[1] == result[2]


def test_reset_clears_visible_and_canonical_history(
    app_module,
) -> None:
    result = app_module.reset_session(request())

    assert result[0] == []
    assert result[1] == []
    assert len(result) == 8
