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


def test_registered_chat_handler_is_the_zero_gpu_boundary(app_module) -> None:
    assert app_module.respond._zero_gpu_config == {
        "size": "large",
        "duration": 90,
    }
    assert not hasattr(app_module.generate_final_answer, "_zero_gpu_config")


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
