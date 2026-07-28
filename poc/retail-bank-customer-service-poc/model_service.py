from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from mock_bank import SessionBankRegistry

INPUT_TOKEN_BUDGET = 8192
MAX_NEW_TOKENS = 512
MAX_TOOL_CALLS = 8

AGENT_SYSTEM_PROMPT = """You are the conversational customer-service agent for a
fictional retail-bank demonstration. All customer records and actions are synthetic.
Respond naturally to greetings, thanks, clarifications, and banking questions. Use
the supplied tools whenever customer-specific backend data or an action is needed.
You own the tool choice and arguments. The classifier information below is advisory:
reason from the conversation and override its predicted intent when appropriate.
If a requested banking capability has no tool, explain that limitation naturally.
After tool results, answer the customer using those results. Do not invent tool
results or claim an action occurred unless a successful tool result says it did."""

MODEL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_accounts",
            "description": "List the signed-in synthetic customer's accounts and balances.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_cards",
            "description": "List the signed-in synthetic customer's cards and statuses.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_service_cases",
            "description": "List recent synthetic service cases, including address changes.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transactions",
            "description": "List recent synthetic account transactions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transfers",
            "description": "List the signed-in synthetic customer's transfers and statuses.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "freeze_card",
            "description": "Freeze a synthetic card, optionally selected by last four digits.",
            "parameters": {
                "type": "object",
                "properties": {"last4": {"type": ["string", "null"]}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_card",
            "description": "Request replacement of a synthetic card.",
            "parameters": {
                "type": "object",
                "properties": {"last4": {"type": ["string", "null"]}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispute_transaction",
            "description": "Dispute a synthetic debit transaction by merchant description.",
            "parameters": {
                "type": "object",
                "properties": {"description": {"type": ["string", "null"]}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_transfer",
            "description": "Cancel a pending synthetic transfer by recipient.",
            "parameters": {
                "type": "object",
                "properties": {"recipient": {"type": ["string", "null"]}},
                "additionalProperties": False,
            },
        },
    },
]

_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    flags=re.DOTALL,
)


class AgentProtocolError(ValueError):
    pass


class AgentExecutionError(AgentProtocolError):
    def __init__(
        self,
        message: str,
        *,
        conversation: list[dict[str, Any]],
        tool_calls: tuple[ToolCall, ...],
        tool_results: tuple[dict[str, Any], ...],
        snapshot: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.conversation = conversation
        self.tool_calls = tool_calls
        self.tool_results = tool_results
        self.snapshot = snapshot


class ModelRuntime(Protocol):
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_new_tokens: int,
    ) -> str:
        ...

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> int:
        ...


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def as_message_call(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass(frozen=True)
class AgentTurnResult:
    response: str
    conversation: list[dict[str, Any]]
    tool_calls: tuple[ToolCall, ...]
    tool_results: tuple[dict[str, Any], ...]
    snapshot: dict[str, Any]


class ConversationalBankingAgent:
    def __init__(
        self,
        *,
        bank: SessionBankRegistry,
        model: ModelRuntime,
        input_budget: int = INPUT_TOKEN_BUDGET,
    ) -> None:
        self.bank = bank
        self.model = model
        self.input_budget = input_budget

    def run_turn(
        self,
        *,
        username: str,
        session_hash: str,
        message: str,
        conversation: list[dict[str, Any]],
        router_result: dict[str, Any],
    ) -> AgentTurnResult:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        canonical = canonical_conversation(conversation)
        current = [*canonical, {"role": "user", "content": message.strip()}]
        system = _system_message(router_result)
        first_context = select_token_budgeted_context(
            system,
            current,
            tools=MODEL_TOOLS,
            token_counter=self.model.count_tokens,
            input_budget=self.input_budget,
        )
        first_output = self.model.generate(
            first_context,
            MODEL_TOOLS,
            MAX_NEW_TOKENS,
        ).strip()
        if not first_output:
            raise AgentProtocolError("model returned an empty first response")
        calls = parse_tool_calls(first_output)
        if not calls:
            completed = [*current, {"role": "assistant", "content": first_output}]
            return AgentTurnResult(
                response=first_output,
                conversation=completed,
                tool_calls=(),
                tool_results=(),
                snapshot=self.bank.snapshot(username, session_hash),
            )

        call_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [call.as_message_call() for call in calls],
        }
        result_messages: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for call in calls:
            result = self._execute_tool(username, session_hash, call)
            results.append(result)
            result_messages.append(
                {
                    "role": "tool",
                    "name": call.name,
                    "content": json.dumps(result, sort_keys=True),
                }
            )
        with_tools = [*current, call_message, *result_messages]
        try:
            second_context = select_token_budgeted_context(
                system,
                with_tools,
                tools=None,
                token_counter=self.model.count_tokens,
                input_budget=self.input_budget,
            )
            final_output = self.model.generate(
                second_context,
                None,
                MAX_NEW_TOKENS,
            ).strip()
            if not final_output:
                raise AgentProtocolError("model returned an empty second response")
            if parse_tool_calls(final_output):
                raise AgentProtocolError("second model response attempted another tool call")
        except (AgentProtocolError, RuntimeError, TypeError, ValueError) as error:
            raise AgentExecutionError(
                str(error),
                conversation=with_tools,
                tool_calls=calls,
                tool_results=tuple(results),
                snapshot=self.bank.snapshot(username, session_hash),
            ) from error
        completed = [*with_tools, {"role": "assistant", "content": final_output}]
        return AgentTurnResult(
            response=final_output,
            conversation=completed,
            tool_calls=calls,
            tool_results=tuple(results),
            snapshot=self.bank.snapshot(username, session_hash),
        )

    def _execute_tool(
        self,
        username: str,
        session_hash: str,
        call: ToolCall,
    ) -> dict[str, Any]:
        try:
            result = self.bank.execute(
                username,
                session_hash,
                call.name,
                call.arguments,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            return {
                "ok": False,
                "name": call.name,
                "error": str(error),
            }
        return {
            "ok": True,
            "name": call.name,
            "result": result,
        }


def parse_tool_calls(output: str) -> tuple[ToolCall, ...]:
    if not isinstance(output, str):
        raise AgentProtocolError("model output must be text")
    blocks = _TOOL_CALL_BLOCK.findall(output)
    has_protocol_marker = "<tool_call" in output or "</tool_call>" in output
    if has_protocol_marker and not blocks:
        raise AgentProtocolError("model returned a malformed tool-call block")
    if len(blocks) > MAX_TOOL_CALLS:
        raise AgentProtocolError(f"model returned more than {MAX_TOOL_CALLS} tool calls")
    calls: list[ToolCall] = []
    for block in blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError as error:
            raise AgentProtocolError("model tool call is not valid JSON") from error
        if not isinstance(payload, dict):
            raise AgentProtocolError("model tool call must be a JSON object")
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not name.strip():
            raise AgentProtocolError("model tool call requires a function name")
        if not isinstance(arguments, dict):
            raise AgentProtocolError("model tool-call arguments must be an object")
        calls.append(ToolCall(name=name.strip(), arguments=arguments))
    return tuple(calls)


def select_token_budgeted_context(
    system_message: dict[str, Any],
    conversation: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    token_counter: Any,
    input_budget: int = INPUT_TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    if input_budget < 1:
        raise ValueError("input budget must be positive")
    canonical = canonical_conversation(conversation)
    groups = _conversation_groups(canonical)
    if not groups:
        selected = [system_message]
        if token_counter(selected, tools) > input_budget:
            raise AgentProtocolError("system prompt exceeds the model input budget")
        return selected

    retained = [groups[-1]]
    selected = [system_message, *groups[-1]]
    if token_counter(selected, tools) > input_budget:
        raise AgentProtocolError("latest conversation turn exceeds the model input budget")
    for group in reversed(groups[:-1]):
        proposal_groups = [group, *retained]
        proposal = [
            system_message,
            *(message for item in proposal_groups for message in item),
        ]
        if token_counter(proposal, tools) <= input_budget:
            retained = proposal_groups
    return [
        system_message,
        *(message for item in retained for message in item),
    ]


def canonical_conversation(
    conversation: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not isinstance(conversation, list):
        return []
    canonical: list[dict[str, Any]] = []
    for item in conversation:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role == "user" and isinstance(content, str) and content.strip():
            canonical.append({"role": "user", "content": content.strip()})
        elif role == "assistant" and isinstance(content, str):
            tool_calls = item.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                canonical.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )
            elif content.strip():
                canonical.append({"role": "assistant", "content": content.strip()})
        elif (
            role == "tool"
            and isinstance(item.get("name"), str)
            and isinstance(content, str)
        ):
            canonical.append(
                {
                    "role": "tool",
                    "name": str(item["name"]),
                    "content": content,
                }
            )
    return canonical


def _conversation_groups(
    conversation: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for item in conversation:
        if item["role"] == "user":
            groups.append([item])
        elif groups:
            groups[-1].append(item)
    return groups


def _system_message(router_result: dict[str, Any]) -> dict[str, str]:
    guidance = {
        "route": router_result.get("route"),
        "banking_probability": router_result.get("banking_probability"),
        "ood_probability": router_result.get("ood_probability"),
        "intent": router_result.get("intent"),
        "intent_confidence": router_result.get("intent_confidence"),
        "intent_candidates": router_result.get("intent_candidates", []),
    }
    return {
        "role": "system",
        "content": (
            f"{AGENT_SYSTEM_PROMPT}\n\n"
            "CURRENT DUAL-HEAD CLASSIFIER GUIDANCE:\n"
            f"{json.dumps(guidance, sort_keys=True)}"
        ),
    }
