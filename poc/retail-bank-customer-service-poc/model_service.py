from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from mock_bank import SessionBankRegistry
from policy import generated_response_is_unsafe

TOOL_SCHEMAS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "list_accounts",
            "description": "List the authenticated synthetic customer's account balances.",
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
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10}
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_cards",
            "description": "List synthetic payment cards and their current status.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transfers",
            "description": "List recent synthetic transfers and their status.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_service_cases",
            "description": "List support cases for the authenticated synthetic customer.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "freeze_card",
            "description": "Freeze a synthetic card and open a synthetic support case.",
            "parameters": {
                "type": "object",
                "properties": {"last4": {"type": "string", "pattern": "^[0-9]{4}$"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_card",
            "description": "Request replacement of a synthetic card and open a case.",
            "parameters": {
                "type": "object",
                "properties": {"last4": {"type": "string", "pattern": "^[0-9]{4}$"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispute_transaction",
            "description": "Dispute a synthetic debit transaction and open a support case.",
            "parameters": {
                "type": "object",
                "properties": {"transaction_id": {"type": "string", "maxLength": 64}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_transfer",
            "description": "Cancel a synthetic transfer only when its status is pending.",
            "parameters": {
                "type": "object",
                "properties": {"transfer_id": {"type": "string", "maxLength": 64}},
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = """You are the conversational agent for a fictional retail-bank demo.
Every customer and banking record is synthetic. For every banking request, call exactly one
provided tool before answering. Never invent balances, identifiers, status, or actions. Never
request a PIN, CVV, password, one-time code, or complete card/account number. Tool actions are
simulations only. Tool results contain display-ready monetary strings; copy those strings
exactly and never recalculate them. After receiving the tool result, answer clearly and mention
that any action was performed only in the synthetic demo."""


class TextGenerator(Protocol):
    def generate(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]] | None,
        max_new_tokens: int,
    ) -> str:
        ...


class ToolCallError(ValueError):
    pass


@dataclass(frozen=True)
class ValidatedToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ServiceReply:
    response: str
    tool_name: str
    tool_result: dict[str, Any]
    snapshot: dict[str, Any]


class ModelDrivenBankingService:
    def __init__(
        self,
        *,
        bank: SessionBankRegistry,
        generator: TextGenerator,
    ) -> None:
        self.bank = bank
        self.generator = generator

    def reply(
        self,
        *,
        username: str,
        session_hash: str,
        message: str,
        history: list[dict[str, Any]],
    ) -> ServiceReply:
        messages = _bounded_messages(message, history)
        selection = self.generator.generate(
            messages,
            tools=TOOL_SCHEMAS,
            max_new_tokens=128,
        )
        tool_call = parse_and_validate_tool_call(selection)
        authorize_tool_call(message, tool_call)
        tool_call = ground_tool_call_arguments(message, tool_call)
        tool_result = self.bank.execute(
            username,
            session_hash,
            tool_call.name,
            tool_call.arguments,
        )
        final_messages = [
            *messages,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": json.dumps(_grounding_payload(tool_result), sort_keys=True),
            },
        ]
        response = self.generator.generate(
            final_messages,
            tools=None,
            max_new_tokens=192,
        ).strip()
        if not response:
            raise ToolCallError("model returned an empty final response")
        if generated_response_is_unsafe(response):
            raise ToolCallError("model requested sensitive credentials")
        return ServiceReply(
            response=response,
            tool_name=tool_call.name,
            tool_result=tool_result,
            snapshot=self.bank.snapshot(username, session_hash),
        )


def parse_and_validate_tool_call(text: str) -> ValidatedToolCall:
    opening = "<tool_call>"
    closing = "</tool_call>"
    if text.count(opening) != 1 or text.count(closing) != 1:
        raise ToolCallError("model must return exactly one tool call")
    payload = text.split(opening, maxsplit=1)[1].split(closing, maxsplit=1)[0].strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ToolCallError("tool call must contain valid JSON") from error
    if not isinstance(parsed, dict) or set(parsed) != {"name", "arguments"}:
        raise ToolCallError("tool call must contain only name and arguments")
    name = parsed["name"]
    arguments = parsed["arguments"]
    if not isinstance(name, str) or name not in _tool_names():
        raise ToolCallError(f"unsupported tool: {name}")
    if not isinstance(arguments, dict):
        raise ToolCallError("tool arguments must be an object")
    allowed = {
        "list_accounts": set(),
        "list_cards": set(),
        "list_service_cases": set(),
        "list_transactions": {"limit"},
        "list_transfers": set(),
        "cancel_transfer": {"transfer_id"},
        "dispute_transaction": {"transaction_id"},
        "freeze_card": {"last4"},
        "replace_card": {"last4"},
    }[name]
    extras = set(arguments) - allowed
    if extras:
        raise ToolCallError(f"unsupported arguments for {name}: {sorted(extras)}")
    if "limit" in arguments:
        limit = arguments["limit"]
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
            raise ToolCallError("transaction limit must be between 1 and 10")
    if "last4" in arguments:
        last4 = arguments["last4"]
        if not isinstance(last4, str) or len(last4) != 4 or not last4.isdigit():
            raise ToolCallError("last4 must contain exactly four digits")
    for identifier in ("transaction_id", "transfer_id"):
        if identifier in arguments:
            value = arguments[identifier]
            if not isinstance(value, str) or not value or len(value) > 64:
                raise ToolCallError(f"{identifier} must be a short non-empty string")
    return ValidatedToolCall(name=name, arguments=arguments)


def authorize_tool_call(message: str, tool_call: ValidatedToolCall) -> None:
    normalized = " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in message).split()
    )
    required_phrases = {
        "freeze_card": ("freeze", "block", "lock", "stolen", "lost"),
        "replace_card": ("replace", "replacement", "new card"),
        "dispute_transaction": (
            "dispute",
            "not mine",
            "did not make",
            "didnt make",
            "unrecognized",
            "unrecognised",
        ),
        "cancel_transfer": ("cancel", "stop", "revoke"),
    }
    phrases = required_phrases.get(tool_call.name)
    if phrases is not None and not any(phrase in normalized for phrase in phrases):
        raise ToolCallError(
            f"{tool_call.name} requires an explicit customer authorization phrase"
        )


def ground_tool_call_arguments(
    message: str,
    tool_call: ValidatedToolCall,
) -> ValidatedToolCall:
    """Discard record selectors the customer did not supply verbatim.

    The model chooses the operation, while the session-scoped backend resolves omitted
    selectors to the only or most recent actionable synthetic record. This prevents a
    hallucinated card suffix or record ID from turning a valid request into the wrong action.
    """

    selector_by_tool = {
        "freeze_card": "last4",
        "replace_card": "last4",
        "dispute_transaction": "transaction_id",
        "cancel_transfer": "transfer_id",
    }
    selector = selector_by_tool.get(tool_call.name)
    if selector is None or selector not in tool_call.arguments:
        return tool_call
    value = tool_call.arguments[selector]
    if isinstance(value, str) and value.lower() in message.lower():
        return tool_call
    arguments = dict(tool_call.arguments)
    arguments.pop(selector)
    return ValidatedToolCall(name=tool_call.name, arguments=arguments)


def _bounded_messages(
    message: str,
    history: list[dict[str, Any]],
) -> list[dict[str, object]]:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    usable = [
        {"role": "user", "content": str(item["content"])}
        for item in history[-8:]
        if isinstance(item, dict)
        and item.get("role") == "user"
        and isinstance(item.get("content"), str)
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *usable,
        {"role": "user", "content": message.strip()},
    ]


def _tool_names() -> set[str]:
    return {
        str(schema["function"]["name"])  # type: ignore[index]
        for schema in TOOL_SCHEMAS
    }


def _grounding_payload(value: Any) -> Any:
    """Replace integer cent fields with exact display-ready currency strings."""

    if isinstance(value, list):
        return [_grounding_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    currency = str(value.get("currency", "USD"))
    grounded: dict[str, Any] = {}
    for key, item in value.items():
        if (
            isinstance(key, str)
            and key.endswith("_cents")
            and isinstance(item, int)
            and not isinstance(item, bool)
        ):
            grounded[key.removesuffix("_cents")] = f"{currency} {item / 100:,.2f}"
        else:
            grounded[str(key)] = _grounding_payload(item)
    return grounded
