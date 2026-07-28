from __future__ import annotations

import json
from pathlib import Path

import pytest

from mock_bank import SessionBankRegistry

ROOT = Path(__file__).parents[1]


def registry() -> SessionBankRegistry:
    return SessionBankRegistry.from_json(ROOT / "synthetic_bank.json", max_sessions=4)


def test_seed_is_explicitly_synthetic_and_contains_two_login_scoped_customers() -> None:
    payload = json.loads((ROOT / "synthetic_bank.json").read_text(encoding="utf-8"))

    assert payload["contract"] == "synthetic-retail-bank-v1"
    assert "fictional" in payload["notice"].lower()
    assert {customer["login"] for customer in payload["customers"]} == {
        "alex.demo",
        "maya.demo",
    }
    assert "password" not in json.dumps(payload).lower()


def test_sessions_and_authenticated_customers_are_isolated() -> None:
    bank = registry()

    bank.execute("alex.demo", "alex-session", "freeze_card", {})

    assert bank.snapshot("alex.demo", "alex-session")["cards"][0]["status"] == "frozen"
    assert bank.snapshot("alex.demo", "second-alex-session")["cards"][0]["status"] == "active"
    assert bank.snapshot("maya.demo", "maya-session")["cards"][0]["last4"] == "7319"


def test_file_backed_state_is_shared_across_worker_registries(tmp_path: Path) -> None:
    first = SessionBankRegistry.from_json(
        ROOT / "synthetic_bank.json",
        database_dir=tmp_path,
    )
    second = SessionBankRegistry.from_json(
        ROOT / "synthetic_bank.json",
        database_dir=tmp_path,
    )

    first.execute("alex.demo", "shared-session", "freeze_card", {})

    assert second.snapshot("alex.demo", "shared-session")["cards"][0]["status"] == "frozen"
    assert second.snapshot("alex.demo", "other-session")["cards"][0]["status"] == "active"

    second.reset("alex.demo", "shared-session")

    assert first.snapshot("alex.demo", "shared-session")["cards"][0]["status"] == "active"


def test_supported_mock_actions_mutate_only_session_database() -> None:
    bank = registry()
    user = "alex.demo"
    session = "action-session"

    disputed = bank.execute(user, session, "dispute_transaction", {})
    cancelled = bank.execute(user, session, "cancel_transfer", {})
    replaced = bank.execute(user, session, "replace_card", {})

    snapshot = bank.snapshot(user, session)
    assert disputed["transaction"]["disputed"] is True
    assert cancelled["transfer"]["status"] == "cancelled"
    assert replaced["card"]["status"] == "replacement_pending"
    assert len(snapshot["service_cases"]) == 3


def test_tool_scope_rejects_unknown_users_sessions_and_cross_customer_arguments() -> None:
    bank = registry()

    with pytest.raises(ValueError, match="unknown authenticated user"):
        bank.snapshot("unknown.demo", "session")
    with pytest.raises(ValueError, match="session hash"):
        bank.snapshot("alex.demo", "")
    with pytest.raises(ValueError, match="unsupported arguments"):
        bank.execute(
            "alex.demo",
            "session",
            "list_accounts",
            {"customer_id": "cust_maya"},
        )
