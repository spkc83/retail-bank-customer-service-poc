from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "DEMO_AUTH_JSON",
        '{"alex.demo":"alex-test-password","maya.demo":"maya-test-password"}',
    )
    monkeypatch.setenv("POC_SKIP_MODEL_LOAD", "1")
    monkeypatch.setenv("POC_SKIP_ROUTER_LOAD", "1")
    for name in ("app", "zero_gpu_runtime", "state"):
        sys.modules.pop(name, None)
    return importlib.import_module("app")


def request(username: str = "alex.demo", session_hash: str = "browser-session"):
    return SimpleNamespace(username=username, session_hash=session_hash)


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
    assert app_module.chat_interface.cache_examples is False


def test_cpu_guards_bypass_zero_gpu_model_call(app_module, monkeypatch) -> None:
    def should_not_run(*_args, **_kwargs):
        raise AssertionError("ZeroGPU must not run")

    monkeypatch.setattr(app_module, "run_model_service", should_not_run)

    sensitive = app_module.respond("My PIN is 1234", [], request())
    ood = app_module.respond("What is the weather?", [], request())

    assert "never needs" in sensitive[0]
    assert "synthetic retail-banking" in ood[0]


def test_model_path_uses_authenticated_identity_and_updates_dashboard(
    app_module,
    monkeypatch,
) -> None:
    class InDomainRouter:
        threshold = 0.98

        def classify(self, message, history):
            return {
                "route": "in_domain",
                "banking_probability": 0.999,
                "intent": "pending_transfer",
            }

    seen = {}

    def model_call(username, session_hash, message, history, intent_hint):
        seen.update(
            {
                "username": username,
                "session_hash": session_hash,
                "message": message,
                "history": history,
                "intent_hint": intent_hint,
            }
        )
        return {
            "response": "I checked the synthetic transfer.",
            "tool_name": "list_transfers",
            "tool_result": {},
            "snapshot": app_module.BANK.snapshot(username, session_hash),
            "model_revision": "revision",
            "selection_source": "model",
        }

    monkeypatch.setattr(app_module, "router", InDomainRouter())
    monkeypatch.setattr(app_module, "_run_model_service", model_call)

    response, dashboard, activity = app_module.respond(
        "Show my transfers.",
        [],
        request("maya.demo", "maya-browser"),
    )

    assert seen["username"] == "maya.demo"
    assert seen["session_hash"] == "maya-browser"
    assert seen["intent_hint"] == "pending_transfer"
    assert "Model tool: `list_transfers`" in response
    assert "Travel Checking" in dashboard
    assert "9B model proposed" in activity


def test_app_reports_learned_intent_repair_without_claiming_model_selection(
    app_module,
    monkeypatch,
) -> None:
    class CancelRouter:
        threshold = 0.98

        def classify(self, message, history):
            return {
                "route": "in_domain",
                "banking_probability": 0.999,
                "intent": "cancel_transfer",
            }

    def repaired_call(username, session_hash, message, history, intent_hint):
        return {
            "response": "I cancelled the USD 450.00 transfer in this synthetic demo.",
            "tool_name": "cancel_transfer",
            "tool_result": {},
            "snapshot": app_module.BANK.snapshot(username, session_hash),
            "model_revision": "revision",
            "selection_source": "router_policy_repair",
        }

    monkeypatch.setattr(app_module, "router", CancelRouter())
    monkeypatch.setattr(app_module, "_run_model_service", repaired_call)

    response, _, activity = app_module.respond(
        "Cancel the transfer of $450 to River Consulting.",
        [],
        request(),
    )

    assert "Model tool: `cancel_transfer`" in response
    assert "learned intent router repaired" in activity
    assert "9B model proposed" not in activity
