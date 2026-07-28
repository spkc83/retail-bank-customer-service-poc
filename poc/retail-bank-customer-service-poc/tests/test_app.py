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
        "orchestration",
        "state",
        "zero_gpu_runtime",
    ):
        sys.modules.pop(name, None)
    return importlib.import_module("app")


def request(username: str = "alex.demo", session_hash: str = "browser-session"):
    return SimpleNamespace(username=username, session_hash=session_hash)


def accepted_route(intent: str) -> dict[str, object]:
    return {
        "route": "in_domain",
        "banking_probability": 0.999,
        "intent": intent,
        "intent_confidence": 0.8,
    }


def test_app_constructs_expected_authenticated_api_surface(app_module) -> None:
    api_names = {
        dependency.get("api_name")
        for dependency in app_module.demo.config["dependencies"]
    }

    assert {"chat", "route", "customer_snapshot", "reset_demo"} <= api_names
    assert {
        "model_selection_probe",
        "gpu_allocation_probe",
        "model_service_probe",
    }.isdisjoint(api_names)
    assert {username for username, _ in app_module.AUTH_CREDENTIALS} == {
        "alex.demo",
        "maya.demo",
    }


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Hello, how are you?", "customer-service assistant"),
        ("What is the weather tomorrow?", "synthetic retail-banking"),
        ("Can you open a mortgage for me?", "not supported"),
    ],
)
def test_direct_paths_bypass_model_finalizer(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    expected: str,
) -> None:
    def should_not_run(*_args, **_kwargs):
        raise AssertionError("ZeroGPU finalizer must not run")

    monkeypatch.setattr(app_module, "generate_final_answer", should_not_run)

    response, _, activity = app_module.respond(message, [], request())

    assert expected in response
    assert "No backend tool" in activity


def test_only_registered_model_turn_is_the_zero_gpu_boundary(app_module) -> None:
    assert not hasattr(app_module.respond, "_zero_gpu_config")
    assert not hasattr(app_module.dispatch_turn, "_zero_gpu_config")
    assert app_module.finalize_turn._zero_gpu_config == {
        "size": "large",
        "duration": 90,
    }
    assert not hasattr(app_module.generate_final_answer, "_zero_gpu_config")


def test_cpu_dispatch_completes_casual_greeting_without_pending_gpu_turn(
    app_module,
) -> None:
    result = app_module.dispatch_turn("yo, sup ?", [], 0, request())

    visible_history = result[1]
    canonical_history = result[2]
    pending_turn = result[5]
    assert visible_history == canonical_history
    assert visible_history[-1]["role"] == "assistant"
    assert "customer-service assistant" in visible_history[-1]["content"]
    assert pending_turn == pytest.importorskip("gradio").skip()


def test_cpu_dispatch_schedules_supported_read_without_running_finalizer(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def should_not_run(*_args, **_kwargs):
        raise AssertionError("CPU dispatch must not run the ZeroGPU finalizer")

    monkeypatch.setattr(app_module, "generate_final_answer", should_not_run)

    result = app_module.dispatch_turn(
        "ok ok, what transfers are there on my account ?",
        [],
        4,
        request(),
    )

    visible_history = result[1]
    canonical_history = result[2]
    pending_turn = result[5]
    assert canonical_history == []
    assert visible_history[-1]["role"] == "assistant"
    assert "9B model" in visible_history[-1]["content"]
    assert pending_turn["message"] == (
        "ok ok, what transfers are there on my account ?"
    )
    assert pending_turn["history"] == []
    assert pending_turn["epoch"] == 4
    assert pending_turn["turn_id"]
    assert "username" not in pending_turn
    assert "session_hash" not in pending_turn


def test_registered_gpu_turn_replaces_pending_answer_with_grounded_response(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "generate_final_answer",
        lambda *_args: (
            "Your synthetic transfers are USD 450.00 to River Consulting, pending, "
            "and USD 125.00 to Jamie Lee, completed."
        ),
    )
    pending = {
        "turn_id": "turn-1",
        "message": "What transfers are there on my account?",
        "history": [],
        "epoch": 2,
    }

    result = app_module.finalize_turn(pending, 2, [], request())

    visible_history = result[0]
    canonical_history = result[1]
    assert visible_history == canonical_history
    assert "River Consulting" in visible_history[-1]["content"]
    assert "Jamie Lee" in visible_history[-1]["content"]
    assert "Model workflow: `list_transfers`" in visible_history[-1]["content"]


def test_stale_gpu_turn_after_reset_executes_nothing(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def should_not_run(*_args, **_kwargs):
        raise AssertionError("stale turn must not route, execute, or finalize")

    monkeypatch.setattr(app_module, "respond", should_not_run)
    current_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Fresh reset session"},
    ]
    pending = {
        "turn_id": "old-turn",
        "message": "Cancel my pending transfer.",
        "history": [],
        "epoch": 3,
    }

    result = app_module.finalize_turn(pending, 4, current_history, request())

    assert result[0] == current_history
    assert result[1] == current_history
    assert "expired" in result[3].lower()


def test_gpu_allocation_failure_replaces_pending_turn_without_mutation(
    app_module,
) -> None:
    pending = {
        "turn_id": "failed-write",
        "message": "Cancel my pending transfer.",
        "history": [],
        "epoch": 5,
    }

    result = app_module.fail_pending_turn(pending, 5, [], request())

    visible_history = result[0]
    canonical_history = result[1]
    assert visible_history == canonical_history
    assert "could not produce" in visible_history[-1]["content"]
    assert "No synthetic write was committed" in result[3]
    assert "`pending`" in result[2]


def test_gpu_allocation_failure_uses_verified_read_only_fallback(
    app_module,
) -> None:
    pending = {
        "turn_id": "failed-read",
        "message": "What transfers are there on my account?",
        "history": [],
        "epoch": 5,
    }

    result = app_module.fail_pending_turn(pending, 5, [], request())

    response = result[0][-1]["content"]
    assert "River Consulting" in response
    assert "Jamie Lee" in response
    assert "verified CPU read fallback" in response
    assert "ZeroGPU was unavailable" in result[3]


def test_sensitive_guard_bypasses_router_and_zero_gpu(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def should_not_run(*_args, **_kwargs):
        raise AssertionError("downstream service must not run")

    monkeypatch.setattr(app_module, "route_query", should_not_run)
    monkeypatch.setattr(app_module, "generate_final_answer", should_not_run)

    response, _, activity = app_module.respond("My PIN is 1234", [], request())

    assert "never needs" in response
    assert "Credential guard" in activity


def test_wrong_router_cannot_override_deterministic_multi_read_workflow(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: {
            "route": "out_of_domain",
            "banking_probability": 0.31,
            "intent": None,
        },
    )
    seen = {}

    def finalizer(messages, grounded_results, max_new_tokens):
        seen["messages"] = messages
        seen["grounded_results"] = grounded_results
        seen["max_new_tokens"] = max_new_tokens
        return "Here are your synthetic transfers and recent transactions."

    monkeypatch.setattr(app_module, "generate_final_answer", finalizer)

    response, _, activity = app_module.respond(
        "Show transfers and recent transactions.",
        [],
        request(),
    )

    assert set(seen["grounded_results"]) == {
        "list_transfers",
        "list_transactions",
    }
    assert "Model workflow: `list_transfers` + `list_transactions`" in response
    assert "deterministic workflow" in activity


def test_cancel_write_uses_exact_session_record_and_model_finalizer(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: accepted_route("cancel_transfer"),
    )
    monkeypatch.setattr(
        app_module,
        "generate_final_answer",
        lambda *_args: (
            "I cancelled the USD 450.00 River Consulting transfer in this "
            "synthetic demo."
        ),
    )

    response, dashboard, activity = app_module.respond(
        "Cancel pending transfer to River Consulting.",
        [],
        request(),
    )

    assert "cancelled" in response
    assert "River Consulting" in dashboard
    assert "`cancelled`" in dashboard
    assert "one explicit write" in activity


def test_model_unavailability_rolls_back_write(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: accepted_route("freeze_card"),
    )

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("No CUDA GPUs are available")

    monkeypatch.setattr(app_module, "generate_final_answer", unavailable)

    response, dashboard, activity = app_module.respond(
        "Freeze my debit card ending in 4821.",
        [],
        request(),
    )

    assert "could not produce" in response
    assert "`active`" in dashboard
    assert "no synthetic action was committed" in activity


def test_completed_transfer_gets_specific_safe_response(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: accepted_route("cancel_transfer"),
    )

    response, dashboard, _ = app_module.respond(
        "Cancel the completed transfer to Jamie Lee.",
        [],
        request(),
    )

    assert "already completed" in response
    assert "cannot be cancelled" in response
    assert "`completed`" in dashboard


def test_mixed_read_write_request_never_calls_model_or_backend(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: accepted_route("cancel_transfer"),
    )

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("ZeroGPU finalizer must not run")

    monkeypatch.setattr(app_module, "generate_final_answer", should_not_run)

    response, dashboard, activity = app_module.respond(
        "Show my transfers and cancel the River Consulting transfer.",
        [],
        request(),
    )

    assert "one account-changing action" in response
    assert "`pending`" in dashboard
    assert "No backend tool" in activity
