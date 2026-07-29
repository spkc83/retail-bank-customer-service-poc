from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from mock_bank import SessionBankRegistry

INPUT_TOKEN_BUDGET = 8192
MAX_NEW_TOKENS = 512
MAX_TOOL_CALLS = 8
USE_ORIGINAL_MARKER = "<use_original/>"

AGENT_SYSTEM_PROMPT = """You are the conversational customer-service agent for a
fictional retail-bank demonstration. All customer records and actions are synthetic.
The customer is already authenticated, and every tool is automatically scoped to
that signed-in customer. Never ask for an account number, customer ID, password,
PIN, or additional identity verification. Respond naturally to greetings, thanks,
clarifications, and general banking questions. Before answering any request about
the customer's accounts, balances, cards, transactions, transfers, service cases,
or an account action, you must call the appropriate supplied tool. Do not answer
customer-specific questions from memory or assumptions. You own the tool choice and
arguments. The classifier information below is advisory: reason from the full
conversation and override its predicted intent when appropriate. If a requested
banking capability has no tool, explain that limitation naturally. After tool
results, answer the customer using those results. Do not invent tool results or
claim an action occurred unless a successful tool result says it did."""

REFLECTION_PROMPT = """You are a second-pass tool-use reviewer for the same
authenticated synthetic-bank conversation. Audit the base draft in the final
TOOL_USE_REVIEW_REQUEST against its customer request and the supplied tool schemas.
This is a decision pass, not the customer-facing answer.

Return exactly one of:
1. <use_original/> when the base draft is appropriate without customer-specific
   backend data or action, including greetings, thanks, general explanations, and
   necessary clarifying questions.
2. One or more Qwen <tool_call> JSON blocks when the base draft should have used a
   supplied tool. Do not add prose around tool calls.

Examples:
- Greeting, with a friendly base draft: <use_original/>
- "How many accounts do I have?", with a draft asking for an account number:
  <tool_call>{"name":"list_accounts","arguments":{}}</tool_call>
- General question about how savings interest works: <use_original/>

Do not copy an example merely because it is present. Decide from the complete
conversation. Never invent a tool and never ask the customer for an account number,
customer ID, password, PIN, or identity verification."""

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


@dataclass(frozen=True)
class ModelPassTrace:
    label: str
    input_tokens: int
    prompt_sha256: str
    output_chars: int
    output_sha256: str
    raw_output: str
    runtime_device: str
    cuda_device_name: str


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
        model_passes: tuple[ModelPassTrace, ...],
    ) -> None:
        super().__init__(message)
        self.conversation = conversation
        self.tool_calls = tool_calls
        self.tool_results = tool_results
        self.snapshot = snapshot
        self.model_passes = model_passes


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
    response_path: str
    model_passes: tuple[ModelPassTrace, ...]


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
        model_passes: list[ModelPassTrace] = []
        first_output, first_trace = self._generate_pass(
            "base",
            first_context,
            MODEL_TOOLS,
        )
        model_passes.append(first_trace)
        if not first_output:
            raise AgentProtocolError("model returned an empty first response")
        calls = parse_tool_calls(first_output)
        if not calls:
            reflection_messages = [
                *current,
                {"role": "assistant", "content": first_output},
                {
                    "role": "user",
                    "content": _reflection_request(message, first_output),
                },
            ]
            reflection_context = select_token_budgeted_context(
                _reflection_system_message(router_result),
                reflection_messages,
                tools=MODEL_TOOLS,
                token_counter=self.model.count_tokens,
                input_budget=self.input_budget,
            )
            reflection_output, reflection_trace = self._generate_pass(
                "reflection",
                reflection_context,
                MODEL_TOOLS,
            )
            model_passes.append(reflection_trace)
            if reflection_output == USE_ORIGINAL_MARKER:
                return self._complete_without_tools(
                    username=username,
                    session_hash=session_hash,
                    current=current,
                    first_output=first_output,
                    response_path="reflection_use_original",
                    model_passes=model_passes,
                )
            try:
                calls = parse_tool_calls(reflection_output)
            except AgentProtocolError:
                calls = ()
            if not calls:
                return self._complete_without_tools(
                    username=username,
                    session_hash=session_hash,
                    current=current,
                    first_output=first_output,
                    response_path="reflection_invalid_use_original",
                    model_passes=model_passes,
                )
            response_path = "reflection_tool"
        else:
            response_path = "base_tool"

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
            final_output, final_trace = self._generate_pass(
                "grounded_final",
                second_context,
                None,
            )
            model_passes.append(final_trace)
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
                model_passes=tuple(model_passes),
            ) from error
        completed = [*with_tools, {"role": "assistant", "content": final_output}]
        return AgentTurnResult(
            response=final_output,
            conversation=completed,
            tool_calls=calls,
            tool_results=tuple(results),
            snapshot=self.bank.snapshot(username, session_hash),
            response_path=response_path,
            model_passes=tuple(model_passes),
        )

    def _complete_without_tools(
        self,
        *,
        username: str,
        session_hash: str,
        current: list[dict[str, Any]],
        first_output: str,
        response_path: str,
        model_passes: list[ModelPassTrace],
    ) -> AgentTurnResult:
        completed = [*current, {"role": "assistant", "content": first_output}]
        return AgentTurnResult(
            response=first_output,
            conversation=completed,
            tool_calls=(),
            tool_results=(),
            snapshot=self.bank.snapshot(username, session_hash),
            response_path=response_path,
            model_passes=tuple(model_passes),
        )

    def _generate_pass(
        self,
        label: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str, ModelPassTrace]:
        input_tokens = self.model.count_tokens(messages, tools)
        prompt_payload = json.dumps(
            {"messages": messages, "tools": tools},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        output = self.model.generate(
            messages,
            tools,
            MAX_NEW_TOKENS,
        ).strip()
        metadata_provider = getattr(self.model, "runtime_metadata", None)
        metadata = metadata_provider() if callable(metadata_provider) else {}
        return output, ModelPassTrace(
            label=label,
            input_tokens=input_tokens,
            prompt_sha256=hashlib.sha256(
                prompt_payload.encode("utf-8")
            ).hexdigest(),
            output_chars=len(output),
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            raw_output=output,
            runtime_device=str(metadata.get("runtime_device", "unavailable")),
            cuda_device_name=str(metadata.get("cuda_device_name", "unavailable")),
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
                "status": "error",
                "action_completed": False,
                "name": call.name,
                "error": str(error),
                "response_requirement": (
                    "The requested action was not completed. Tell the customer it "
                    "failed and explain this error; do not claim success."
                ),
            }
        return {
            "ok": True,
            "status": "success",
            "action_completed": True,
            "name": call.name,
            "result": _model_friendly_result(result),
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


def _model_friendly_result(value: Any, currency: str | None = None) -> Any:
    if isinstance(value, dict):
        local_currency = (
            str(value["currency"])
            if isinstance(value.get("currency"), str)
            else currency
        )
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if (
                key.endswith("_cents")
                and isinstance(item, int)
                and not isinstance(item, bool)
            ):
                amount_key = key.removesuffix("_cents")
                currency_code = local_currency or "USD"
                normalized[amount_key] = f"{currency_code} {item / 100:,.2f}"
            else:
                normalized[key] = _model_friendly_result(item, local_currency)
        return normalized
    if isinstance(value, list):
        return [_model_friendly_result(item, currency) for item in value]
    return value


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


def _reflection_system_message(
    router_result: dict[str, Any],
) -> dict[str, str]:
    base_system = _system_message(router_result)["content"]
    return {
        "role": "system",
        "content": (
            f"{base_system}\n\n"
            f"{REFLECTION_PROMPT}"
        ),
    }


def _reflection_request(message: str, first_output: str) -> str:
    payload = {
        "customer_request": message.strip(),
        "base_draft": first_output,
    }
    return (
        "TOOL_USE_REVIEW_REQUEST\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n"
        "Return only <use_original/> or valid Qwen <tool_call> blocks."
    )
