from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mock_bank import SessionBankRegistry
from model_service import (
    GroundedBankingService,
    ModelResponseError,
    _bounded_messages,
    _grounding_payload,
    verified_read_response,
)
from orchestration import plan_workflow

ROOT = Path(__file__).parents[1]


def bank() -> SessionBankRegistry:
    return SessionBankRegistry.from_json(ROOT / "synthetic_bank.json")


def route(*, in_domain: bool = True, intent: str | None = None) -> dict[str, Any]:
    return {
        "route": "in_domain" if in_domain else "out_of_domain",
        "intent": intent,
        "banking_probability": 0.99 if in_domain else 0.2,
    }


class RecordingFinalizer:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        messages: list[dict[str, str]],
        grounded_results: dict[str, Any],
        max_new_tokens: int,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "grounded_results": grounded_results,
                "max_new_tokens": max_new_tokens,
            }
        )
        return self.outputs.pop(0)


def test_verified_read_response_rejects_write_results() -> None:
    with pytest.raises(ValueError, match="read-only"):
        verified_read_response(
            {
                "cancel_transfer": {
                    "transfer": {
                        "recipient": "River Consulting",
                        "amount_cents": 45000,
                        "status": "cancelled",
                    }
                }
            }
        )


def test_multi_read_executes_exact_workflow_and_model_finalizes_grounded_bundle() -> None:
    finalizer = RecordingFinalizer(
        ["Your recent transfers and transactions are shown from the synthetic demo."]
    )
    service = GroundedBankingService(bank=bank(), finalizer=finalizer)
    message = "Show transfers and recent transactions."
    plan = plan_workflow(message, [], route(intent="pending_transfer"))

    reply = service.execute(
        username="alex.demo",
        session_hash="session",
        message=message,
        history=[],
        plan=plan,
    )

    assert reply.workflow_tools == ("list_transfers", "list_transactions")
    assert reply.selection_source == "grounded_repair"
    assert len(finalizer.calls) == 1
    grounded = finalizer.calls[0]["grounded_results"]
    assert tuple(grounded) == ("list_transfers", "list_transactions")
    serialized = json.dumps(grounded)
    assert "_id" not in serialized
    assert "_cents" not in serialized
    assert "USD 450.00" in serialized


def test_single_read_uses_service_cases_for_address_history() -> None:
    finalizer = RecordingFinalizer(
        [
            "The available synthetic service-case history shows the mailing address "
            "update on June 18, 2026."
        ]
    )
    service = GroundedBankingService(bank=bank(), finalizer=finalizer)
    message = "When was my mailing address changed?"

    reply = service.execute(
        username="alex.demo",
        session_hash="session",
        message=message,
        history=[],
        plan=plan_workflow(message, [], route(intent="edit_personal_details")),
    )

    assert reply.workflow_tools == ("list_service_cases",)
    grounded = finalizer.calls[0]["grounded_results"]
    assert grounded["list_service_cases"]["service_cases"][0]["created_at"].startswith(
        "2026-06-18"
    )


def test_cancel_pending_transfer_resolves_exact_recipient_and_commits_after_finalization() -> None:
    finalizer = RecordingFinalizer(
        ["I cancelled the USD 450.00 River Consulting transfer in the synthetic demo."]
    )
    registry = bank()
    service = GroundedBankingService(bank=registry, finalizer=finalizer)
    message = "Cancel pending transfer to River Consulting."

    reply = service.execute(
        username="alex.demo",
        session_hash="session",
        message=message,
        history=[],
        plan=plan_workflow(message, [], route(in_domain=False)),
    )

    assert reply.workflow_tools == ("cancel_transfer",)
    assert reply.tool_result["cancel_transfer"]["transfer"]["status"] == "cancelled"
    assert registry.snapshot("alex.demo", "session")["transfers"][0]["status"] == "cancelled"


def test_cancel_without_selector_resolves_only_when_one_pending_transfer_exists() -> None:
    finalizer = RecordingFinalizer(
        ["I cancelled the only pending transfer in this synthetic demo."]
    )
    registry = bank()
    service = GroundedBankingService(bank=registry, finalizer=finalizer)
    message = "Cancel my pending transfer."

    reply = service.execute(
        username="alex.demo",
        session_hash="session",
        message=message,
        history=[],
        plan=plan_workflow(message, [], route()),
    )

    assert reply.tool_result["cancel_transfer"]["transfer"]["recipient"] == "River Consulting"
    assert reply.tool_result["cancel_transfer"]["transfer"]["status"] == "cancelled"


def test_completed_transfer_cancellation_fails_without_mutation_or_model_call() -> None:
    finalizer = RecordingFinalizer(["This output must not be used."])
    registry = bank()
    service = GroundedBankingService(bank=registry, finalizer=finalizer)
    message = "Cancel the completed transfer to Jamie Lee."

    with pytest.raises(ModelResponseError, match="not pending"):
        service.execute(
            username="alex.demo",
            session_hash="session",
            message=message,
            history=[],
            plan=plan_workflow(message, [], route()),
        )

    assert finalizer.calls == []
    assert [item["status"] for item in registry.snapshot("alex.demo", "session")["transfers"]] == [
        "pending",
        "completed",
    ]


def test_unsafe_final_answer_rolls_back_write() -> None:
    finalizer = RecordingFinalizer(
        ["Please provide your password so I can finish freezing the card."]
    )
    registry = bank()
    service = GroundedBankingService(bank=registry, finalizer=finalizer)
    message = "Freeze my debit card ending in 4821."

    with pytest.raises(ModelResponseError, match="unsafe"):
        service.execute(
            username="alex.demo",
            session_hash="session",
            message=message,
            history=[],
            plan=plan_workflow(message, [], route()),
        )

    assert registry.snapshot("alex.demo", "session")["cards"][0]["status"] == "active"


def test_empty_final_answer_rolls_back_write() -> None:
    finalizer = RecordingFinalizer(["  "])
    registry = bank()
    service = GroundedBankingService(bank=registry, finalizer=finalizer)
    message = "Replace my debit card ending in 4821."

    with pytest.raises(ModelResponseError, match="empty"):
        service.execute(
            username="alex.demo",
            session_hash="session",
            message=message,
            history=[],
            plan=plan_workflow(message, [], route()),
        )

    assert registry.snapshot("alex.demo", "session")["cards"][0]["status"] == "active"


def test_internal_identifier_in_final_answer_is_rejected_and_write_rolls_back() -> None:
    finalizer = RecordingFinalizer(
        ["I froze internal record card_alex_debit in the synthetic demo."]
    )
    registry = bank()
    service = GroundedBankingService(bank=registry, finalizer=finalizer)
    message = "Freeze my debit card."

    with pytest.raises(ModelResponseError, match="internal identifier"):
        service.execute(
            username="alex.demo",
            session_hash="session",
            message=message,
            history=[],
            plan=plan_workflow(message, [], route()),
        )

    assert registry.snapshot("alex.demo", "session")["cards"][0]["status"] == "active"


def test_bounded_messages_preserve_sanitized_user_and_assistant_turns() -> None:
    messages = _bounded_messages(
        "What about my card?",
        [
            {"role": "user", "content": "Show my balances."},
            {
                "role": "assistant",
                "content": (
                    "Your synthetic checking balance is USD 12,500.00.\n\n"
                    "---\n_Model workflow: `list_accounts` · revision `abc…`_"
                ),
            },
        ],
    )

    assert {"role": "user", "content": "Show my balances."} in messages
    assert {
        "role": "assistant",
        "content": "Your synthetic checking balance is USD 12,500.00.",
    } in messages
    assert all("Model workflow" not in item["content"] for item in messages)


def test_grounding_payload_removes_internal_fields_and_formats_money() -> None:
    grounded = _grounding_payload(
        {
            "transfer_id": "trf_internal",
            "from_account_id": "acct_internal",
            "login": "alex.demo",
            "recipient": "River Consulting",
            "amount_cents": 45_000,
            "currency": "USD",
        }
    )

    assert grounded == {
        "recipient": "River Consulting",
        "amount": "USD 450.00",
        "currency": "USD",
    }


def test_ungrounded_model_money_is_replaced_with_verified_backend_values() -> None:
    finalizer = RecordingFinalizer(
        ["Your checking account balance is USD 9,999.99."]
    )
    service = GroundedBankingService(bank=bank(), finalizer=finalizer)
    message = "What is my checking account balance?"

    reply = service.execute(
        username="alex.demo",
        session_hash="session",
        message=message,
        history=[],
        plan=plan_workflow(
            message,
            [],
            route(intent="balance_not_updated_after_cheque_or_cash_deposit"),
        ),
    )

    assert reply.selection_source == "grounded_repair"
    assert "USD 3,245.67" in reply.response
    assert "USD 3,300.12 current" in reply.response
    assert "USD 9,999.99" not in reply.response
    assert "All data and actions shown here are synthetic." in reply.response


def test_incomplete_account_balance_answer_is_replaced_with_labeled_values() -> None:
    finalizer = RecordingFinalizer(
        ["Your checking account balance is USD 3,300.12."]
    )
    service = GroundedBankingService(bank=bank(), finalizer=finalizer)
    message = "What is my checking account balance?"

    reply = service.execute(
        username="alex.demo",
        session_hash="session",
        message=message,
        history=[],
        plan=plan_workflow(message, [], route(intent="cash_withdrawal")),
    )

    assert reply.selection_source == "grounded_repair"
    assert "USD 3,245.67 available" in reply.response
    assert "USD 3,300.12 current" in reply.response


def test_misleading_cancel_acknowledgement_is_replaced_before_commit() -> None:
    finalizer = RecordingFinalizer(
        ["Please provide the transfer reference so I can initiate cancellation."]
    )
    registry = bank()
    service = GroundedBankingService(bank=registry, finalizer=finalizer)
    message = "Cancel the pending transfer to River Consulting."

    reply = service.execute(
        username="alex.demo",
        session_hash="session",
        message=message,
        history=[],
        plan=plan_workflow(message, [], route(intent="cancel_transfer")),
    )

    assert reply.selection_source == "grounded_repair"
    assert "USD 450.00 transfer to River Consulting is cancelled" in reply.response
    assert registry.snapshot("alex.demo", "session")["transfers"][0]["status"] == "cancelled"


def test_address_history_without_limit_qualifier_uses_grounded_repair() -> None:
    finalizer = RecordingFinalizer(
        ["Your mailing address was updated on June 18, 2026."]
    )
    service = GroundedBankingService(bank=bank(), finalizer=finalizer)
    message = "When was my mailing address changed?"

    reply = service.execute(
        username="alex.demo",
        session_hash="session",
        message=message,
        history=[],
        plan=plan_workflow(message, [], route(intent="edit_personal_details")),
    )

    assert reply.selection_source == "grounded_repair"
    assert "Limited service-case history" in reply.response
    assert "2026-06-18" in reply.response
