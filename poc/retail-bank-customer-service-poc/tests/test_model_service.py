from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mock_bank import SessionBankRegistry
from model_service import (
    INPUT_TOKEN_BUDGET,
    MODEL_TOOLS,
    AgentExecutionError,
    AgentProtocolError,
    ConversationalBankingAgent,
    parse_tool_calls,
    select_token_budgeted_context,
)

ROOT = Path(__file__).parents[1]


def bank() -> SessionBankRegistry:
    return SessionBankRegistry.from_json(ROOT / "synthetic_bank.json")


class RecordingModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
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

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> int:
        return len(json.dumps({"messages": messages, "tools": tools}))


def router_guidance() -> dict[str, Any]:
    return {
        "route": "in_domain",
        "banking_probability": 0.99,
        "ood_probability": 0.01,
        "intent": "pending_transfer",
        "intent_confidence": 0.81,
        "intent_candidates": [
            {"intent": "pending_transfer", "probability": 0.81},
            {"intent": "cancel_transfer", "probability": 0.12},
            {"intent": "card_payment_fee_charged", "probability": 0.03},
        ],
    }


def test_public_tool_schemas_use_customer_facing_arguments() -> None:
    schemas = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in MODEL_TOOLS
    }

    assert set(schemas) == {
        "list_accounts",
        "list_cards",
        "list_service_cases",
        "list_transactions",
        "list_transfers",
        "freeze_card",
        "replace_card",
        "dispute_transaction",
        "cancel_transfer",
    }
    assert set(schemas["cancel_transfer"]["properties"]) == {"recipient"}
    assert set(schemas["dispute_transaction"]["properties"]) == {"description"}
    assert "transfer_id" not in json.dumps(MODEL_TOOLS)
    assert "transaction_id" not in json.dumps(MODEL_TOOLS)


def test_tagged_json_parser_accepts_multiple_ordered_calls_and_ignores_prose() -> None:
    calls = parse_tool_calls(
        """I will check both.
<tool_call>
{"name": "list_transfers", "arguments": {}}
</tool_call>
<tool_call>
{"name": "list_transactions", "arguments": {"limit": 3}}
</tool_call>"""
    )

    assert [call.name for call in calls] == ["list_transfers", "list_transactions"]
    assert calls[0].id.startswith("call_")
    assert calls[0].id.endswith("_0_list_transfers")
    assert calls[1].id.endswith("_1_list_transactions")
    assert calls[0].id != calls[1].id
    assert [call.index for call in calls] == [0, 1]
    assert calls[1].arguments == {"limit": 3}


@pytest.mark.parametrize(
    "output",
    [
        "<tool_call>not-json</tool_call>",
        '<tool_call>{"name": "", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "list_accounts", "arguments": []}</tool_call>',
        "<tool_call>",
    ],
)
def test_tagged_json_parser_rejects_malformed_protocol(output: str) -> None:
    with pytest.raises(AgentProtocolError):
        parse_tool_calls(output)


def test_plain_first_pass_text_is_a_model_authored_conversational_answer() -> None:
    model = RecordingModel(
        [
            "Hey! What can I help you with today?",
            "<use_original/>",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="yo, sup?",
        conversation=[],
        router_result=router_guidance(),
    )

    assert result.response == "Hey! What can I help you with today?"
    assert result.tool_calls == ()
    assert result.conversation == [
        {"role": "user", "content": "yo, sup?"},
        {"role": "assistant", "content": result.response},
    ]
    assert result.response_path == "reflection_use_original"
    assert [item.label for item in result.model_passes] == ["base", "reflection"]
    assert result.model_passes[0].raw_output == result.response
    assert len(model.calls) == 2
    assert model.calls[0]["tools"] == MODEL_TOOLS
    assert model.calls[1]["tools"] == MODEL_TOOLS


def test_reflection_can_recover_a_missing_tool_call_without_hiding_base_output() -> None:
    model = RecordingModel(
        [
            "Please provide your account number so I can check.",
            '<tool_call>{"name": "list_accounts", "arguments": {}}</tool_call>',
            "You have checking and savings accounts.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="How many accounts do I have?",
        conversation=[],
        router_result=router_guidance(),
    )

    assert result.response == "You have checking and savings accounts."
    assert result.response_path == "reflection_tool"
    assert [item.label for item in result.model_passes] == [
        "base",
        "reflection",
        "grounded_final",
    ]
    assert (
        result.model_passes[0].raw_output
        == "Please provide your account number so I can check."
    )
    assert [call.name for call in result.tool_calls] == ["list_accounts"]
    assert [item["role"] for item in result.conversation] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.conversation[1]["tool_calls"][0]["id"].endswith("_0_list_accounts")
    assert (
        result.conversation[2]["tool_call_id"]
        == result.conversation[1]["tool_calls"][0]["id"]
    )
    assert len(model.calls) == 3
    reflection_messages = model.calls[1]["messages"]
    assert reflection_messages[-1]["role"] == "user"
    assert "TOOL_USE_REVIEW_REQUEST" in reflection_messages[-1]["content"]
    assert "How many accounts do I have?" in reflection_messages[-1]["content"]
    assert "Please provide your account number" in reflection_messages[-1]["content"]


def test_invalid_reflection_output_cannot_approve_the_base_answer() -> None:
    model = RecordingModel(
        [
            "I can explain how savings interest works.",
            "I think the draft is fine.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    with pytest.raises(
        AgentProtocolError,
        match="must be <use_original/> or a valid tool call",
    ):
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="How does savings interest work?",
            conversation=[],
            router_result=router_guidance(),
        )
    assert len(model.calls) == 2


def test_malformed_reflection_cannot_approve_customer_specific_no_tool_answer() -> None:
    model = RecordingModel(
        [
            "Your checking account balance is $1,000.",
            "<tool_call>not-json</tool_call>",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    with pytest.raises(AgentProtocolError, match="malformed tool-call syntax"):
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="What is my checking account balance?",
            conversation=[],
            router_result=router_guidance(),
        )


def test_tool_calls_execute_in_order_and_second_model_pass_writes_final_answer() -> None:
    model = RecordingModel(
        [
            """
<tool_call>
{"name": "list_transfers", "arguments": {}}
</tool_call>
<tool_call>
{"name": "list_transactions", "arguments": {"limit": 2}}
</tool_call>
""",
            "You have two transfers. River Consulting is pending, and Jamie Lee is completed.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="What transfers are there, and show my latest transactions too.",
        conversation=[],
        router_result=router_guidance(),
    )

    assert [call.name for call in result.tool_calls] == [
        "list_transfers",
        "list_transactions",
    ]
    assert len(result.tool_results) == 2
    assert all(set(item) == {"ok", "result"} for item in result.tool_results)
    assert all(item["ok"] is True for item in result.tool_results)
    assert len(model.calls) == 2
    assert model.calls[0]["tools"] == MODEL_TOOLS
    assert model.calls[1]["tools"] == MODEL_TOOLS
    assert result.tool_calls[0].id.endswith("_0_list_transfers")
    assert result.tool_calls[1].id.endswith("_1_list_transactions")
    assert model.calls[1]["messages"][-2]["tool_call_id"] == result.tool_calls[0].id
    assert model.calls[1]["messages"][-1]["tool_call_id"] == result.tool_calls[1].id
    transfer_result = model.calls[1]["messages"][-2]["content"]
    assert json.loads(transfer_result) == result.tool_results[0]
    assert '"amount_cents":45000' in transfer_result
    assert '"amount":' not in transfer_result
    system_prompt = model.calls[0]["messages"][0]["content"]
    assert "already authenticated" in system_prompt
    assert "must call the appropriate supplied tool" in system_prompt
    assert "Never ask for an account number" in system_prompt
    assert [item["role"] for item in result.conversation] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert result.conversation[-1]["content"] == result.response


def test_model_selected_write_uses_friendly_argument_without_authorization_layer() -> None:
    model = RecordingModel(
        [
            """
<tool_call>
{"name": "cancel_transfer", "arguments": {"recipient": "River Consulting"}}
</tool_call>
""",
            "Done — I cancelled the River Consulting transfer.",
        ]
    )
    registry = bank()
    agent = ConversationalBankingAgent(bank=registry, model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Please take care of the River Consulting transfer.",
        conversation=[],
        router_result=router_guidance(),
    )

    assert result.tool_results[0]["ok"] is True
    assert set(result.tool_results[0]) == {"ok", "result"}
    assert registry.snapshot("alex.demo", "session")["transfers"][0]["status"] == "cancelled"


def test_backend_error_returns_safe_canonical_tool_result_to_model() -> None:
    model = RecordingModel(
        [
            """
<tool_call>
{"name": "cancel_transfer", "arguments": {"recipient": "Nobody"}}
</tool_call>
""",
            "I could not complete that operation because I could not find a matching transfer.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Do those operations.",
        conversation=[],
        router_result=router_guidance(),
    )

    assert result.tool_results == (
        {
            "ok": False,
            "error": {
                "code": "record_match_count",
                "message": "The request did not match exactly one synthetic banking record.",
            },
        },
    )
    assert result.conversation[2]["tool_call_id"].endswith("_0_cancel_transfer")
    assert json.loads(model.calls[1]["messages"][-1]["content"]) == result.tool_results[0]
    assert len(model.calls) == 2


def test_invalid_model_arguments_remain_protocol_failures_without_repair() -> None:
    model = RecordingModel(
        [
            '<tool_call>{"name": "list_transactions", "arguments": {"limit": "two"}}</tool_call>',
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    with pytest.raises(AgentProtocolError, match="invalid type"):
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Show my latest two transactions.",
            conversation=[],
            router_result=router_guidance(),
        )

    assert len(model.calls) == 1


def test_unknown_model_tool_remains_protocol_failure_without_fallback() -> None:
    model = RecordingModel(
        [
            '<tool_call>{"name": "close_account", "arguments": {}}</tool_call>',
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    with pytest.raises(AgentProtocolError, match="unsupported tool"):
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Close my account.",
            conversation=[],
            router_result=router_guidance(),
        )

    assert len(model.calls) == 1


def test_repeated_same_name_calls_keep_distinct_tool_call_ids() -> None:
    model = RecordingModel(
        [
            """
<tool_call>
{"name": "list_transactions", "arguments": {"limit": 1}}
</tool_call>
<tool_call>
{"name": "list_transactions", "arguments": {"limit": 2}}
</tool_call>
""",
            "Here are the requested transaction views.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Show my latest transaction, then show my latest two transactions.",
        conversation=[],
        router_result=router_guidance(),
    )

    assert [call.name for call in result.tool_calls] == [
        "list_transactions",
        "list_transactions",
    ]
    assert result.tool_calls[0].id.endswith("_0_list_transactions")
    assert result.tool_calls[1].id.endswith("_1_list_transactions")
    assert result.tool_calls[0].id != result.tool_calls[1].id
    assert [
        item["tool_call_id"]
        for item in model.calls[1]["messages"]
        if item["role"] == "tool"
    ] == [call.id for call in result.tool_calls]
    assert all(item["ok"] is True for item in result.tool_results)


def test_fallback_tool_call_ids_do_not_collide_across_retained_turns() -> None:
    tool_output = '<tool_call>{"name": "list_transfers", "arguments": {}}</tool_call>'
    model = RecordingModel(
        [
            tool_output,
            "You have a pending River Consulting transfer.",
            tool_output,
            "You still have a pending River Consulting transfer.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    first = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Show my transfers.",
        conversation=[],
        router_result=router_guidance(),
    )
    second = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Show my transfers again.",
        conversation=first.conversation,
        router_result=router_guidance(),
    )

    first_id = first.tool_calls[0].id
    second_id = second.tool_calls[0].id
    assert first_id.endswith("_0_list_transfers")
    assert second_id.endswith("_0_list_transfers")
    assert first_id != second_id
    retained_tool_ids = [
        item["tool_call_id"]
        for item in second.conversation
        if item["role"] == "tool"
    ]
    assert retained_tool_ids == [first_id, second_id]


def test_duplicate_model_tool_call_ids_are_protocol_failures() -> None:
    model = RecordingModel(
        [
            """
<tool_call>
{"id": "call_duplicate", "index": 0, "name": "list_accounts", "arguments": {}}
</tool_call>
<tool_call>
{"id": "call_duplicate", "index": 1, "name": "list_cards", "arguments": {}}
</tool_call>
""",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    with pytest.raises(AgentProtocolError, match="IDs must be unique"):
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Show my accounts and cards.",
            conversation=[],
            router_result=router_guidance(),
        )

    assert len(model.calls) == 1


def test_second_pass_tool_call_is_a_protocol_error_after_tool_already_executed() -> None:
    model = RecordingModel(
        [
            '<tool_call>{"name": "freeze_card", "arguments": {"last4": "4821"}}</tool_call>',
            '<tool_call>{"name": "list_cards", "arguments": {}}</tool_call>',
        ]
    )
    registry = bank()
    agent = ConversationalBankingAgent(bank=registry, model=model)

    with pytest.raises(AgentExecutionError, match="second") as failure:
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Freeze card 4821.",
            conversation=[],
            router_result=router_guidance(),
        )

    assert failure.value.tool_calls[0].name == "freeze_card"
    assert failure.value.tool_results[0]["ok"] is True
    assert [item["role"] for item in failure.value.conversation] == [
        "user",
        "assistant",
        "tool",
    ]
    assert registry.snapshot("alex.demo", "session")["cards"][0]["status"] == "frozen"


def test_token_budget_keeps_latest_complete_tool_chain_and_newest_fitting_turns() -> None:
    system = {"role": "system", "content": "system"}
    old = [
        {"role": "user", "content": "old " * 30},
        {"role": "assistant", "content": "old answer " * 30},
    ]
    middle = [
        {"role": "user", "content": "middle"},
        {"role": "assistant", "content": "middle answer"},
    ]
    latest = [
        {"role": "user", "content": "show transfers"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_context_0_list_transfers",
                    "index": 0,
                    "type": "function",
                    "function": {"name": "list_transfers", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_context_0_list_transfers",
            "name": "list_transfers",
            "content": '{"ok": true}',
        },
    ]

    selected = select_token_budgeted_context(
        system,
        [*old, *middle, *latest],
        tools=MODEL_TOOLS,
        token_counter=lambda messages, _tools: len(json.dumps(messages)),
        input_budget=540,
    )

    assert selected[0] == system
    assert latest == selected[-len(latest) :]
    assert middle[0] in selected
    assert old[0] not in selected
    latest_roles = [item["role"] for item in selected[-len(latest) :]]
    assert latest_roles == ["user", "assistant", "tool"]


def test_token_budget_rejects_oversized_latest_group_without_truncation() -> None:
    with pytest.raises(AgentProtocolError, match="latest conversation turn"):
        select_token_budgeted_context(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "x" * 100}],
            tools=MODEL_TOOLS,
            token_counter=lambda messages, _tools: len(json.dumps(messages)),
            input_budget=50,
        )


def test_default_input_budget_is_8192_tokens() -> None:
    assert INPUT_TOKEN_BUDGET == 8192
