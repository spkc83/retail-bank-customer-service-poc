from __future__ import annotations

from pathlib import Path

import pytest

from mock_bank import SessionBankRegistry
from model_service import (
    ModelDrivenBankingService,
    ToolCallError,
    ValidatedToolCall,
    _grounding_payload,
    authorize_tool_call,
    ground_tool_call_arguments,
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


def test_hallucinated_selector_is_removed_before_backend_execution() -> None:
    generator = ScriptedGenerator(
        [
            (
                '<tool_call>{"name":"freeze_card",'
                '"arguments":{"last4":"1234"}}</tool_call>'
            ),
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

    assert reply.snapshot["cards"][0]["last4"] == "4821"
    assert reply.snapshot["cards"][0]["status"] == "frozen"


def test_prior_model_output_is_not_replayed_into_tool_selection() -> None:
    generator = ScriptedGenerator(
        [
            '<tool_call>{"name":"freeze_card","arguments":{}}</tool_call>',
            "I froze the synthetic debit card.",
        ]
    )
    service = ModelDrivenBankingService(bank=bank(), generator=generator)

    service.reply(
        username="alex.demo",
        session_hash="session",
        message="My card was stolen. Freeze it.",
        history=[
            {"role": "user", "content": "Show my balances."},
            {
                "role": "assistant",
                "content": "Previous model answer with tool metadata.",
            },
        ],
    )

    selection_messages = generator.calls[0]["messages"]
    assert isinstance(selection_messages, list)
    assert {"role": "user", "content": "Show my balances."} in selection_messages
    assert all(
        message.get("content") != "Previous model answer with tool metadata."
        for message in selection_messages
        if isinstance(message, dict)
    )


def test_customer_supplied_selector_is_preserved() -> None:
    tool_call = ValidatedToolCall(name="freeze_card", arguments={"last4": "4821"})

    assert ground_tool_call_arguments("Freeze card 4821.", tool_call) == tool_call


def test_ungrounded_selector_is_removed() -> None:
    tool_call = ValidatedToolCall(name="freeze_card", arguments={"last4": "1234"})

    assert ground_tool_call_arguments("Freeze my stolen card.", tool_call) == ValidatedToolCall(
        name="freeze_card",
        arguments={},
    )


def test_grounding_payload_formats_money_without_leaving_raw_cents() -> None:
    grounded = _grounding_payload(
        {
            "accounts": [
                {
                    "currency": "USD",
                    "available_balance_cents": 1_250_000,
                    "current_balance_cents": 1_250_000,
                }
            ]
        }
    )

    assert grounded == {
        "accounts": [
            {
                "currency": "USD",
                "available_balance": "USD 12,500.00",
                "current_balance": "USD 12,500.00",
            }
        ]
    }


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


def test_unsafe_final_response_is_rejected_after_read_only_tool() -> None:
    service = ModelDrivenBankingService(
        bank=bank(),
        generator=ScriptedGenerator(
            [
                '<tool_call>{"name":"list_accounts","arguments":{}}</tool_call>',
                "Please provide your password so I can continue.",
            ]
        ),
    )

    with pytest.raises(ToolCallError, match="sensitive credentials"):
        service.reply(
            username="alex.demo",
            session_hash="session",
            message="Show my balances.",
            history=[],
        )


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
