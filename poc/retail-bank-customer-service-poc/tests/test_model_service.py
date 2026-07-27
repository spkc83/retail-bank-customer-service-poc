from __future__ import annotations

from pathlib import Path

import pytest

from mock_bank import SessionBankRegistry
from model_service import (
    ModelDrivenBankingService,
    ToolCallError,
    ValidatedToolCall,
    authorize_tool_call,
    parse_and_validate_tool_call,
)

ROOT = Path(__file__).parents[1]


class ScriptedGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]] | None,
        max_new_tokens: int,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "max_new_tokens": max_new_tokens,
            }
        )
        return self.outputs.pop(0)


def bank() -> SessionBankRegistry:
    return SessionBankRegistry.from_json(ROOT / "synthetic_bank.json")


def test_model_tool_call_executes_mock_action_then_model_writes_final_answer() -> None:
    generator = ScriptedGenerator(
        [
            '<tool_call>{"name":"freeze_card","arguments":{}}</tool_call>',
            "I froze the synthetic debit card ending in 4821.",
        ]
    )
    service = ModelDrivenBankingService(bank=bank(), generator=generator)

    reply = service.reply(
        username="alex.demo",
        session_hash="session",
        message="My card was stolen. Freeze it.",
        history=[],
    )

    assert reply.tool_name == "freeze_card"
    assert "4821" in reply.response
    assert reply.snapshot["cards"][0]["status"] == "frozen"
    assert len(generator.calls) == 2
    assert generator.calls[0]["tools"]
    assert generator.calls[1]["tools"] is None


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ("No tool needed", "exactly one tool call"),
        (
            '<tool_call>{"name":"wire_money","arguments":{}}</tool_call>',
            "unsupported tool",
        ),
        (
            '<tool_call>{"name":"list_accounts","arguments":{"customer_id":"cust_maya"}}</tool_call>',
            "unsupported arguments",
        ),
        (
            '<tool_call>{"name":"list_transactions","arguments":{"limit":500}}</tool_call>',
            "between 1 and 10",
        ),
    ],
)
def test_tool_call_validation_rejects_unstructured_unsupported_or_unsafe_calls(
    payload: str,
    error: str,
) -> None:
    with pytest.raises(ToolCallError, match=error):
        parse_and_validate_tool_call(payload)


def test_model_failure_does_not_mutate_database() -> None:
    registry = bank()
    service = ModelDrivenBankingService(
        bank=registry,
        generator=ScriptedGenerator(["I can do that without a tool."]),
    )

    with pytest.raises(ToolCallError):
        service.reply(
            username="maya.demo",
            session_hash="session",
            message="Cancel my transfer.",
            history=[],
        )

    assert registry.snapshot("maya.demo", "session")["transfers"][0]["status"] == "completed"


@pytest.mark.parametrize(
    ("message", "tool_name"),
    [
        ("My card was stolen; freeze it.", "freeze_card"),
        ("Please replace my debit card.", "replace_card"),
        ("I did not make the latest card purchase. Dispute it.", "dispute_transaction"),
        ("Cancel my pending transfer.", "cancel_transfer"),
    ],
)
def test_write_tools_require_concrete_customer_authorization_phrases(
    message: str,
    tool_name: str,
) -> None:
    authorize_tool_call(message, ValidatedToolCall(name=tool_name, arguments={}))

    with pytest.raises(ToolCallError, match="explicit customer authorization"):
        authorize_tool_call(
            "Tell me about it.",
            ValidatedToolCall(name=tool_name, arguments={}),
        )
