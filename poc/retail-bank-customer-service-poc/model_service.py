from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from mock_bank import SessionBankRegistry
from orchestration import WorkflowPlan
from policy import generated_response_is_unsafe

FINALIZER_SYSTEM_PROMPT = """You are the customer-service response writer for a
fictional retail-bank demo. The server has already selected and executed the
permitted workflow. Answer the customer's current request using only the supplied
verified workflow results. Do not invent balances, dates, identifiers, status, or
actions. Never request a PIN, CVV, password, one-time code, or complete card/account
number. Monetary strings are display-ready and must be copied exactly. Do not expose
internal identifiers. For account-balance requests, state both available_balance and
current_balance with explicit labels. If service cases are supplied, describe them as
the limited history available in this POC. Keep the answer clear, concise,
conversational, and state that data and actions are synthetic."""

_INTERNAL_IDENTIFIER = re.compile(
    r"\b(?:acct|card|case|cust|trf|txn)_[A-Za-z0-9_]+\b",
    re.IGNORECASE,
)
_DISPLAY_MONEY = re.compile(r"\b[A-Z]{3}\s+-?[0-9][0-9,]*\.[0-9]{2}\b")


class FinalAnswerGenerator(Protocol):
    def __call__(
        self,
        messages: list[dict[str, str]],
        grounded_results: dict[str, Any],
        max_new_tokens: int,
    ) -> str:
        ...


class ModelResponseError(ValueError):
    pass


@dataclass(frozen=True)
class ServiceReply:
    response: str
    workflow_tools: tuple[str, ...]
    tool_result: dict[str, Any]
    snapshot: dict[str, Any]
    selection_source: str = "deterministic_workflow"


@dataclass(frozen=True)
class FinalizedAnswer:
    response: str
    selection_source: str


class GroundedBankingService:
    def __init__(
        self,
        *,
        bank: SessionBankRegistry,
        finalizer: FinalAnswerGenerator,
    ) -> None:
        self.bank = bank
        self.finalizer = finalizer

    def execute(
        self,
        *,
        username: str,
        session_hash: str,
        message: str,
        history: list[dict[str, Any]],
        plan: WorkflowPlan,
    ) -> ServiceReply:
        messages = _bounded_messages(message, history)
        if plan.category in {"single_read", "multi_read"}:
            calls = tuple(
                (
                    tool,
                    {"limit": 5} if tool == "list_transactions" else {},
                )
                for tool in plan.read_tools
            )
            raw_results = self.bank.execute_read_bundle(
                username,
                session_hash,
                calls,
            )
            finalized = self._finalize(messages, raw_results)
            return ServiceReply(
                response=finalized.response,
                workflow_tools=plan.read_tools,
                tool_result=raw_results,
                snapshot=self.bank.snapshot(username, session_hash),
                selection_source=finalized.selection_source,
            )

        if plan.category == "single_write" and plan.write_tool is not None:
            _authorize_write(message, plan.write_tool)
            try:
                arguments = _resolve_write_arguments(
                    message,
                    plan.write_tool,
                    plan.arguments,
                    self.bank.snapshot(username, session_hash),
                )

                def finalize(result: dict[str, Any]) -> FinalizedAnswer:
                    return self._finalize(messages, {plan.write_tool: result})

                result, finalized = self.bank.execute_atomic(
                    username,
                    session_hash,
                    plan.write_tool,
                    arguments,
                    finalize=finalize,
                )
            except ModelResponseError:
                raise
            except ValueError as error:
                raise ModelResponseError(str(error)) from error
            return ServiceReply(
                response=finalized.response,
                workflow_tools=(plan.write_tool,),
                tool_result={plan.write_tool: result},
                snapshot=self.bank.snapshot(username, session_hash),
                selection_source=finalized.selection_source,
            )

        raise ValueError(f"workflow category {plan.category} does not execute tools")

    def _finalize(
        self,
        messages: list[dict[str, str]],
        raw_results: dict[str, Any],
    ) -> FinalizedAnswer:
        grounded_results = _grounding_payload(raw_results)
        response = self.finalizer(messages, grounded_results, 256).strip()
        if not response:
            raise ModelResponseError("model returned an empty final response")
        if generated_response_is_unsafe(response):
            raise ModelResponseError("model returned an unsafe final response")
        if _contains_internal_identifier(response, raw_results):
            raise ModelResponseError("model exposed an internal identifier")
        if _contains_ungrounded_money(
            response,
            grounded_results,
        ) or _violates_account_balance_contract(response, grounded_results):
            return FinalizedAnswer(
                response=_deterministic_grounded_response(grounded_results),
                selection_source="grounded_repair",
            )
        return FinalizedAnswer(
            response=response,
            selection_source="model_finalizer",
        )


def _bounded_messages(
    message: str,
    history: list[dict[str, Any]],
    *,
    max_turn_messages: int = 8,
) -> list[dict[str, str]]:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    usable: list[dict[str, str]] = []
    for item in history[-max_turn_messages:]:
        if (
            not isinstance(item, dict)
            or item.get("role") not in {"user", "assistant"}
            or not isinstance(item.get("content"), str)
        ):
            continue
        content = str(item["content"]).strip()
        if item["role"] == "assistant":
            content = _sanitize_assistant_content(content)
        if content:
            usable.append({"role": str(item["role"]), "content": content})
    return [
        {"role": "system", "content": FINALIZER_SYSTEM_PROMPT},
        *usable,
        {"role": "user", "content": message.strip()},
    ]


def _sanitize_assistant_content(content: str) -> str:
    return content.split("\n\n---\n_", maxsplit=1)[0].strip()


def _authorize_write(message: str, tool: str) -> None:
    normalized = _normalize(message)
    phrases = {
        "cancel_transfer": ("cancel", "stop", "revoke"),
        "freeze_card": ("freeze", "block", "lock"),
        "replace_card": ("replace", "replacement"),
        "dispute_transaction": (
            "dispute",
            "not mine",
            "did not make",
            "didnt make",
            "unrecognized",
            "unrecognised",
        ),
    }[tool]
    if not any(phrase in normalized for phrase in phrases):
        raise ModelResponseError(f"{tool} lacks explicit customer authorization")


def _resolve_write_arguments(
    message: str,
    tool: str,
    proposed: dict[str, object],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    if tool in {"freeze_card", "replace_card"}:
        return _resolve_card_arguments(proposed, snapshot)
    if tool == "cancel_transfer":
        return _resolve_transfer_arguments(message, snapshot)
    if tool == "dispute_transaction":
        return _resolve_transaction_arguments(message, snapshot)
    raise ModelResponseError(f"unsupported write workflow: {tool}")


def _resolve_card_arguments(
    proposed: dict[str, object],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    cards = [item for item in snapshot.get("cards", []) if isinstance(item, dict)]
    last4 = proposed.get("last4")
    if isinstance(last4, str):
        cards = [item for item in cards if item.get("last4") == last4]
    if len(cards) != 1:
        raise ModelResponseError("identify exactly one synthetic card by its last four digits")
    resolved = cards[0].get("last4")
    if not isinstance(resolved, str):
        raise ModelResponseError("the selected synthetic card is invalid")
    return {"last4": resolved}


def _resolve_transaction_arguments(
    message: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    transactions = [
        item
        for item in snapshot.get("transactions", [])
        if isinstance(item, dict)
        and int(item.get("amount_cents", 0)) < 0
        and item.get("disputed") is not True
    ]
    normalized = _normalize(message)
    named = [
        item
        for item in transactions
        if _normalize(str(item.get("description", ""))) in normalized
    ]
    if len(named) == 1:
        transactions = named
    elif re.search(r"\b(?:latest|last|most recent)\b", normalized):
        transactions = transactions[:1]
    else:
        raise ModelResponseError(
            "identify one transaction by merchant description or ask to dispute the latest purchase"
        )
    if len(transactions) != 1:
        raise ModelResponseError("identify exactly one synthetic transaction")
    transaction_id = transactions[0].get("transaction_id")
    if not isinstance(transaction_id, str):
        raise ModelResponseError("the selected synthetic transaction is invalid")
    return {"transaction_id": transaction_id}


def _resolve_transfer_arguments(
    message: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    transfers = [
        item for item in snapshot.get("transfers", []) if isinstance(item, dict)
    ]
    recipient = _recipient_query(message)
    amount_cents = _amount_query_cents(message)
    described = transfers
    if recipient is not None:
        described = [
            item
            for item in described
            if _normalize(str(item.get("recipient", ""))) == recipient
        ]
    if amount_cents is not None:
        described = [
            item for item in described if item.get("amount_cents") == amount_cents
        ]
    if recipient is None and amount_cents is None:
        pending = [item for item in described if item.get("status") == "pending"]
        if len(pending) != 1:
            raise ModelResponseError("identify one pending transfer by recipient or amount")
        transfer_id = pending[0].get("transfer_id")
        if not isinstance(transfer_id, str):
            raise ModelResponseError("the selected synthetic transfer is invalid")
        return {"transfer_id": transfer_id}
    if len(described) == 1 and described[0].get("status") != "pending":
        raise ModelResponseError("the described transfer is not pending")
    pending = [item for item in described if item.get("status") == "pending"]
    if len(pending) != 1:
        raise ModelResponseError("identify exactly one pending synthetic transfer")
    transfer_id = pending[0].get("transfer_id")
    if not isinstance(transfer_id, str):
        raise ModelResponseError("the selected synthetic transfer is invalid")
    return {"transfer_id": transfer_id}


def _recipient_query(message: str) -> str | None:
    match = re.search(
        (
            r"\btransfer\b.*?\bto\s+(.+?)"
            r"(?=\s+(?:for|of)\s+(?:\$\s*|USD\s+)?[0-9]"
            r"|\s+USD\s+[0-9]|[.!?]|$)"
        ),
        message,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    recipient = _normalize(match.group(1))
    return recipient or None


def _amount_query_cents(message: str) -> int | None:
    match = re.search(
        r"(?:\$\s*|\bUSD\s+|\b(?:of|for)\s+)([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
        message,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    whole, separator, fraction = match.group(1).replace(",", "").partition(".")
    cents = int(whole) * 100
    return cents + (int(fraction.ljust(2, "0")) if separator else 0)


def _contains_internal_identifier(response: str, raw_results: dict[str, Any]) -> bool:
    if _INTERNAL_IDENTIFIER.search(response):
        return True
    response_lower = response.lower()
    return any(
        value.lower() in response_lower
        for value in _internal_values(raw_results)
        if len(value) >= 6
    )


def _contains_ungrounded_money(
    response: str,
    grounded_results: dict[str, Any],
) -> bool:
    displayed = set(_DISPLAY_MONEY.findall(response))
    if not displayed:
        return False
    allowed = set(_DISPLAY_MONEY.findall(str(grounded_results)))
    return not displayed <= allowed


def _violates_account_balance_contract(
    response: str,
    grounded_results: dict[str, Any],
) -> bool:
    payload = grounded_results.get("list_accounts")
    if not isinstance(payload, dict):
        return False
    accounts = payload.get("accounts")
    if not isinstance(accounts, list):
        return False
    required_values = {
        str(item[key])
        for item in accounts
        if isinstance(item, dict)
        for key in ("available_balance", "current_balance")
        if isinstance(item.get(key), str)
    }
    normalized = response.lower()
    return (
        "available" not in normalized
        or "current" not in normalized
        or not required_values <= set(_DISPLAY_MONEY.findall(response))
    )


def _deterministic_grounded_response(grounded_results: dict[str, Any]) -> str:
    lines = ["Here are the verified results from this synthetic banking demo:"]
    for tool, payload in grounded_results.items():
        if not isinstance(payload, dict):
            continue
        if tool == "list_accounts":
            lines.extend(
                f"- {item['name']} ending {item['last4']}: "
                f"{item['available_balance']} available; "
                f"{item['current_balance']} current ({item['status']})."
                for item in payload.get("accounts", [])
            )
        elif tool == "list_cards":
            lines.extend(
                f"- {item['name']} ending {item['last4']}: {item['status']}."
                for item in payload.get("cards", [])
            )
        elif tool == "list_transactions":
            lines.extend(
                f"- {item['posted_at'][:10]} — {item['description']}: "
                f"{item['amount']} ({item['status']})."
                for item in payload.get("transactions", [])
            )
        elif tool == "list_transfers":
            lines.extend(
                f"- {item['recipient']}: {item['amount']} ({item['status']})."
                for item in payload.get("transfers", [])
            )
        elif tool == "list_service_cases":
            lines.append("- Limited service-case history available in this POC:")
            lines.extend(
                f"  - {item['created_at'][:10]} — {item['subject']} ({item['status']})."
                for item in payload.get("service_cases", [])
            )
        elif tool in {"freeze_card", "replace_card"}:
            item = payload.get("card", {})
            lines.append(
                f"- Card ending {item['last4']} is now {item['status']}."
            )
        elif tool == "dispute_transaction":
            item = payload.get("transaction", {})
            lines.append(
                f"- The {item['description']} transaction for {item['amount']} "
                "is now marked as disputed."
            )
        elif tool == "cancel_transfer":
            item = payload.get("transfer", {})
            lines.append(
                f"- The {item['amount']} transfer to {item['recipient']} is "
                f"{item['status']}."
            )
    lines.append("All data and actions shown here are synthetic.")
    return "\n".join(lines)


def _internal_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for entry in value for item in _internal_values(entry)]
    if not isinstance(value, dict):
        return []
    values: list[str] = []
    for key, item in value.items():
        if isinstance(key, str) and (key.endswith("_id") or key == "login"):
            if isinstance(item, str):
                values.append(item)
        else:
            values.extend(_internal_values(item))
    return values


def _grounding_payload(value: Any) -> Any:
    if isinstance(value, list):
        return [_grounding_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    currency = str(value.get("currency", "USD"))
    grounded: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(key, str) and (key.endswith("_id") or key == "login"):
            continue
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


def _normalize(text: str) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in text)
        .split()
    )
