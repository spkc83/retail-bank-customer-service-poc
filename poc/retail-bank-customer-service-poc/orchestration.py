from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from policy import OOD_RESPONSE

WorkflowCategory = Literal[
    "conversational",
    "single_read",
    "multi_read",
    "single_write",
    "unsupported_banking",
    "out_of_domain",
    "clarification",
]

GREETING_RESPONSE = (
    "Hello! I’m the customer-service assistant for this synthetic retail-bank demo. "
    "I can help with the signed-in demo customer’s balances, cards, transactions, "
    "transfers, and service cases."
)
ACK_RESPONSE = (
    "You’re welcome. I’m ready to help with another request for this synthetic "
    "retail-bank demo."
)
UNSUPPORTED_BANKING_RESPONSE = (
    "That is a banking request, but that service is not supported by this POC. I can "
    "help with the signed-in synthetic customer’s balances, cards, transactions, "
    "transfers, or service cases."
)
CLARIFICATION_RESPONSE = (
    "For safety, please ask for one account-changing action at a time and do not "
    "combine it with another request. No synthetic data was changed."
)


@dataclass(frozen=True)
class WorkflowPlan:
    category: WorkflowCategory
    read_tools: tuple[str, ...]
    write_tool: str | None
    arguments: dict[str, object]
    direct_response: str | None
    router_intent: str | None
    router_route: str
    reason: str


@dataclass(frozen=True)
class _DetectedWrite:
    tool: str
    arguments: dict[str, object]
    position: int


_READ_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "list_accounts",
        (
            re.compile(r"\b(?:account\s+)?balances?\b"),
            re.compile(r"\b(?:available|current)\s+balance\b"),
        ),
    ),
    (
        "list_transactions",
        (
            re.compile(r"\b(?:recent|latest|last|show|list|what)\b.{0,32}\btransactions?\b"),
            re.compile(r"\btransactions?\b.{0,32}\b(?:recent|latest|last|show|list|status)\b"),
            re.compile(r"\brecent\s+(?:account\s+)?activity\b"),
            re.compile(r"\b(?:recent|latest|last|show|list)\b.{0,32}\b(?:payments?|purchases?|spending)\b"),
        ),
    ),
    (
        "list_cards",
        (
            re.compile(
                r"\b(?:card|cards|debit card|credit card)\b.{0,32}"
                r"\b(?:status|working|active|show|list)\b"
            ),
            re.compile(r"\b(?:status|show|list)\b.{0,32}\b(?:card|cards|debit card|credit card)\b"),
        ),
    ),
    (
        "list_transfers",
        (
            re.compile(
                r"\b(?:show|list|recent|latest|last|pending|status)\b"
                r".{0,32}\btransfers?\b"
            ),
            re.compile(r"\b(?:what|which)\b.{0,32}\btransfers\b"),
            re.compile(r"\btransfers?\b.{0,32}\b(?:show|list|recent|latest|last|pending|status)\b"),
        ),
    ),
    (
        "list_service_cases",
        (
            re.compile(r"\b(?:show|list|open|recent)\b.{0,32}\b(?:service\s+)?cases?\b"),
            re.compile(r"\b(?:mailing\s+)?address\b.{0,32}\b(?:changed|change|updated|update|history|when)\b"),
            re.compile(r"\bwhen\b.{0,32}\b(?:mailing\s+)?address\b"),
            re.compile(r"\b(?:support|service)\s+(?:request|case)\b"),
        ),
    ),
)

_BANKING_TERMS = {
    "account",
    "atm",
    "bank",
    "banking",
    "beneficiary",
    "cash",
    "card",
    "deposit",
    "exchange",
    "fee",
    "funds",
    "iban",
    "identity",
    "loan",
    "mortgage",
    "payment",
    "pin",
    "refund",
    "statement",
    "swift",
    "topup",
    "transaction",
    "transfer",
    "withdrawal",
}

_OOD_TERMS = {
    "forecast",
    "movie",
    "restaurant",
    "sports",
    "temperature",
    "weather",
}


def plan_workflow(
    message: str,
    history: list[dict[str, Any]],
    router_result: dict[str, Any],
) -> WorkflowPlan:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    router_route = str(router_result.get("route", "out_of_domain"))
    router_intent_value = router_result.get("intent")
    router_intent = (
        str(router_intent_value) if isinstance(router_intent_value, str) else None
    )
    normalized = _normalize(message)
    terms = set(normalized.split())

    conversational = _conversational_response(normalized, terms)
    if conversational is not None:
        return _direct(
            "conversational",
            conversational,
            router_route,
            router_intent,
            "narrow greeting or acknowledgement",
        )

    if terms & _OOD_TERMS:
        return _direct(
            "out_of_domain",
            OOD_RESPONSE,
            router_route,
            router_intent,
            "explicit non-banking subject",
        )

    context = _recent_context(history)
    writes = _detect_writes(normalized, context)
    reads = _detect_reads(normalized)
    reads = _remove_incidental_write_reads(normalized, writes, reads)

    if len(writes) > 1 or (writes and reads):
        return _direct(
            "clarification",
            CLARIFICATION_RESPONSE,
            router_route,
            router_intent,
            "mixed or multiple account-changing operations",
        )

    if len(writes) == 1:
        write = writes[0]
        return WorkflowPlan(
            category="single_write",
            read_tools=(),
            write_tool=write.tool,
            arguments=write.arguments,
            direct_response=None,
            router_intent=router_intent,
            router_route=router_route,
            reason="explicit customer write request matched a supported workflow",
        )

    if reads:
        tools = tuple(tool for _, tool in sorted(reads, key=lambda item: item[0]))
        return WorkflowPlan(
            category="single_read" if len(tools) == 1 else "multi_read",
            read_tools=tools,
            write_tool=None,
            arguments={},
            direct_response=None,
            router_intent=router_intent,
            router_route=router_route,
            reason="explicit customer read request matched supported workflow evidence",
        )

    if terms & _BANKING_TERMS:
        return _direct(
            "unsupported_banking",
            UNSUPPORTED_BANKING_RESPONSE,
            router_route,
            router_intent,
            "banking request has no implemented backend workflow",
        )

    return _direct(
        "out_of_domain",
        OOD_RESPONSE,
        router_route,
        router_intent,
        "no explicit supported banking workflow evidence",
    )


def _conversational_response(normalized: str, terms: set[str]) -> str | None:
    if terms & (_BANKING_TERMS | _OOD_TERMS):
        return None
    if re.fullmatch(
        r"(?:hello|hi|hey|good morning|good afternoon|good evening)"
        r"(?: how are you| there| everyone)?",
        normalized,
    ):
        return GREETING_RESPONSE
    if re.fullmatch(
        r"(?:yo(?: sup| what s up)?|sup|what s up|how s it going)",
        normalized,
    ):
        return GREETING_RESPONSE
    if re.fullmatch(
        r"(?:thanks|thank you|thank you very much|great thanks|okay thanks|ok thanks)",
        normalized,
    ):
        return ACK_RESPONSE
    return None


def _detect_reads(normalized: str) -> list[tuple[int, str]]:
    detected: list[tuple[int, str]] = []
    for tool, patterns in _READ_PATTERNS:
        positions = [
            match.start()
            for pattern in patterns
            if (match := pattern.search(normalized)) is not None
        ]
        if positions:
            detected.append((_read_subject_position(normalized, tool), tool))
    return detected


def _read_subject_position(normalized: str, tool: str) -> int:
    subjects = {
        "list_accounts": ("balance", "account"),
        "list_transactions": ("transaction", "activity", "payment", "purchase", "spending"),
        "list_cards": ("card",),
        "list_transfers": ("transfer",),
        "list_service_cases": ("address", "case", "support", "service"),
    }[tool]
    positions = [
        position
        for subject in subjects
        if (position := normalized.find(subject)) >= 0
    ]
    return min(positions) if positions else len(normalized)


def _remove_incidental_write_reads(
    normalized: str,
    writes: list[_DetectedWrite],
    reads: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    if not writes:
        return reads
    explicit_read = re.search(
        r"\b(?:show|list|display|what|status|recent activity)\b",
        normalized,
    )
    if explicit_read is not None:
        return reads
    incidental = {
        "cancel_transfer": "list_transfers",
        "freeze_card": "list_cards",
        "replace_card": "list_cards",
        "dispute_transaction": "list_transactions",
    }
    suppressed = {incidental[write.tool] for write in writes}
    return [item for item in reads if item[1] not in suppressed]


def _detect_writes(normalized: str, context: str) -> list[_DetectedWrite]:
    writes: list[_DetectedWrite] = []
    combined = f"{normalized} {context}".strip()

    cancel_match = re.search(r"\b(?:cancel|stop|revoke)\b", normalized)
    if cancel_match is not None and re.search(r"\btransfer\b", combined):
        writes.append(_DetectedWrite("cancel_transfer", {}, cancel_match.start()))

    freeze_match = re.search(r"\b(?:freeze|block|lock)\b", normalized)
    if freeze_match is not None and re.search(r"\bcard\b", combined):
        writes.append(
            _DetectedWrite(
                "freeze_card",
                _card_arguments(normalized),
                freeze_match.start(),
            )
        )

    replace_match = re.search(r"\b(?:replace|replacement)\b", normalized)
    if replace_match is not None and re.search(r"\bcard\b", combined):
        writes.append(
            _DetectedWrite(
                "replace_card",
                _card_arguments(normalized),
                replace_match.start(),
            )
        )

    dispute_match = re.search(
        r"\b(?:dispute|not mine|did not make|didnt make|unrecogni[sz]ed)\b",
        normalized,
    )
    if dispute_match is not None and re.search(
        r"\b(?:transaction|payment|purchase)\b",
        combined,
    ):
        writes.append(
            _DetectedWrite("dispute_transaction", {}, dispute_match.start())
        )

    return sorted(writes, key=lambda item: item.position)


def _card_arguments(normalized: str) -> dict[str, object]:
    match = re.search(r"\b(?:ending(?: in)?|last four|last4)\s+([0-9]{4})\b", normalized)
    return {"last4": match.group(1)} if match is not None else {}


def _recent_context(history: list[dict[str, Any]]) -> str:
    for item in reversed(history[-8:]):
        if (
            isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
        ):
            content = _normalize(str(item["content"]).split("\n\n---\n", maxsplit=1)[0])
            if content:
                return content
    return ""


def _direct(
    category: WorkflowCategory,
    response: str,
    router_route: str,
    router_intent: str | None,
    reason: str,
) -> WorkflowPlan:
    return WorkflowPlan(
        category=category,
        read_tools=(),
        write_tool=None,
        arguments={},
        direct_response=response,
        router_intent=router_intent,
        router_route=router_route,
        reason=reason,
    )


def _normalize(text: str) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in text)
        .split()
    )
