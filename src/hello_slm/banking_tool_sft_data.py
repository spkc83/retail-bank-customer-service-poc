from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hello_slm.config import canonical_json_bytes, file_sha256

BANKING_TOOL_SFT_CONTRACT = "banking-tool-sft/v1"
BANKING_TOOL_SFT_MANIFEST_CONTRACT = "banking-tool-sft-manifest"
CREATED_AT = "2026-07-29T00:00:00Z"
GENERATOR_VERSION = "banking-tool-sft/v1.3-servicing-quality"
DEFAULT_OUTPUT_DIR = Path("data/banking-v3-tool-sft")
DEFAULT_SYNTHETIC_BANK_PATH = Path("poc/retail-bank-customer-service-poc/synthetic_bank.json")
SPLITS = ("train", "validation", "test")
ALLOWED_ARGS = {
    "list_accounts": set(),
    "list_cards": set(),
    "list_service_cases": set(),
    "list_transactions": {"limit"},
    "list_transfers": set(),
    "cancel_transfer": {"recipient"},
    "dispute_transaction": {"description"},
    "freeze_card": {"last4"},
    "replace_card": {"last4"},
}
READ_TOOLS = {
    "list_accounts",
    "list_cards",
    "list_service_cases",
    "list_transactions",
    "list_transfers",
}
WRITE_TOOLS = set(ALLOWED_ARGS) - READ_TOOLS
SYSTEM_PROMPT = (
    "You are the conversational customer-service agent for a fictional retail-bank "
    "demonstration. The customer is already authenticated. Use the supplied tools for "
    "customer-specific banking records or actions, use tool results for final answers, "
    "call dependent tools one at a time so each later call can use the earlier result, "
    "and never ask for account numbers, customer IDs, passwords, PINs, or private IDs."
)
NO_TOOL_OOD_RESPONSE = (
    "I can only help with retail banking and financial-services questions. Please ask "
    "about accounts, cards, transfers, payments, loans, or related banking support."
)
NORMALIZED_TEXT_RE = re.compile(r"[^a-z0-9]+")
REALIZER_OPENERS = (
    "Please",
    "Can you",
    "I need you to",
    "Could you",
    "Help me",
    "Would you",
    "I want to",
    "Before I leave,",
    "In the mobile app context,",
    "For my banking session,",
    "When you have a moment,",
    "For this signed-in profile,",
)
REALIZER_CLOSERS = (
    "",
    "today",
    "right now",
    "for my records",
    "before I continue",
    "in this chat",
    "without asking for private IDs",
    "using the secure banking tools",
    "based on my signed-in profile",
    "and keep it concise",
)
REALIZER_CONTEXTS = (
    "",
    "because I am reviewing my monthly budget",
    "before I make another payment",
    "while I am checking the mobile app",
    "so I can decide what to do next",
    "because I am reconciling recent activity",
    "before I leave for a trip",
    "while I am updating my records",
    "because I noticed something unexpected",
    "so I can finish this banking task",
)
FICTIONAL_FIRST_NAMES = (
    "Alex",
    "Maya",
    "Jordan",
    "Taylor",
    "Morgan",
    "Riley",
    "Casey",
    "Avery",
    "Quinn",
    "Rowan",
)
FICTIONAL_LAST_NAMES = (
    "Morgan",
    "Chen",
    "Patel",
    "Rivera",
    "Brooks",
    "Nguyen",
    "Carter",
    "Singh",
    "Bennett",
    "Reed",
)
ACCOUNT_LABELS = (
    "Everyday Checking",
    "Main Checking",
    "Household Checking",
    "Campus Checking",
    "Travel Checking",
    "Freelance Checking",
    "Shared Bills Checking",
    "Primary Checking",
)
SAVINGS_LABELS = (
    "Goal Saver",
    "Emergency Savings",
    "Holiday Savings",
    "Reserve Savings",
    "Rainy Day Savings",
    "Trip Savings",
    "Home Fund",
    "Safety Net Savings",
)
CARD_LABELS = (
    "Everyday Visa Debit",
    "Travel Visa Debit",
    "Household Debit",
    "Campus Debit",
    "Primary Debit",
    "Market Debit",
    "Reserve Debit",
    "Bills Debit",
)
MERCHANT_PREFIXES = (
    "North Harbor",
    "Pine Ridge",
    "Cedar Point",
    "Maple Street",
    "Summit Lane",
    "Lakeview",
    "Oak Hollow",
    "Silver Creek",
    "Bright Meadow",
    "Riverbend",
)
MERCHANT_TYPES = (
    "Market",
    "Books",
    "Pharmacy",
    "Cafe",
    "Hardware",
    "Transit",
    "Fitness",
    "Bakery",
    "Electronics",
    "Florist",
)
RECIPIENT_PREFIXES = (
    "River",
    "Harbor",
    "Summit",
    "Cedar",
    "Maple",
    "Prairie",
    "Lake",
    "Oak",
    "Bright",
    "Silver",
)
RECIPIENT_TYPES = (
    "Consulting",
    "Rentals",
    "Studios",
    "Design",
    "Services",
    "Repair",
    "Landscaping",
    "Tuition",
    "Utilities",
    "Catering",
)
REALIZER_FINAL_PREFIXES = ("",)
FAQ_REQUIRED_MARKERS = {
    "faq-overdraft-v1": ("overdraft",),
    "faq-mortgage-opening-v1": ("mortgage", "cannot open"),
    "faq-mortgage-age-v1": ("mortgage", "18", "cannot determine"),
    "faq-deposit-opening-v1": ("account", "cannot open"),
    "faq-savings-interest-v1": ("interest",),
}


class BankingToolSftDataError(ValueError):
    """Raised when banking-v3 tool-use records fail generation or validation."""


@dataclass(frozen=True)
class ToolPlan:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    scenario_family: str
    template_id: str
    realization_seed: str
    customer_login: str
    customer_id: str
    state_seed: str
    user: str
    final_response: str
    path: str
    tool_plan: tuple[ToolPlan, ...] = ()
    pre_messages: tuple[dict[str, Any], ...] = ()
    grounding_facts: tuple[str, ...] = ()
    bank_payload: dict[str, Any] | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare banking-v3 tool-use SFT data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pilot-count", type=int, default=1200)
    parser.add_argument("--split-seed", type=int, default=7303)
    parser.add_argument("--synthetic-bank", type=Path, default=DEFAULT_SYNTHETIC_BANK_PATH)
    parser.add_argument(
        "--export-teacher-requests",
        type=Path,
        help="Write teacher realization request JSONL for the generated semantic records.",
    )
    parser.add_argument(
        "--teacher-responses",
        type=Path,
        help="Apply teacher response JSONL before writing split files.",
    )
    parser.add_argument("--teacher-model", help="Teacher model ID used for --teacher-responses.")
    parser.add_argument(
        "--teacher-prompt-hash",
        help="Teacher prompt hash used for --teacher-responses.",
    )
    args = parser.parse_args(argv)
    try:
        report = prepare(
            output_dir=args.output_dir,
            pilot_count=args.pilot_count,
            split_seed=args.split_seed,
            synthetic_bank_path=args.synthetic_bank,
            export_teacher_requests=args.export_teacher_requests,
            teacher_responses=args.teacher_responses,
            teacher_model=args.teacher_model,
            teacher_prompt_hash=args.teacher_prompt_hash,
        )
    except (BankingToolSftDataError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "success", **report["summary"]}, sort_keys=True))
    return 0


def public_tool_manifest() -> list[dict[str, Any]]:
    return [
        _tool("list_accounts", "List the signed-in synthetic customer's accounts and balances."),
        _tool("list_cards", "List the signed-in synthetic customer's cards and statuses."),
        _tool("list_service_cases", "List recent synthetic service cases."),
        _tool(
            "list_transactions",
            "List recent synthetic account transactions.",
            {"limit": {"type": "integer", "minimum": 1, "maximum": 20}},
        ),
        _tool("list_transfers", "List the signed-in synthetic customer's transfers and statuses."),
        _tool(
            "freeze_card",
            "Freeze a synthetic card, optionally selected by last four digits.",
            {"last4": {"type": ["string", "null"]}},
        ),
        _tool(
            "replace_card",
            "Request replacement of a synthetic card.",
            {"last4": {"type": ["string", "null"]}},
        ),
        _tool(
            "dispute_transaction",
            "Dispute a synthetic transaction by description.",
            {"description": {"type": ["string", "null"]}},
        ),
        _tool(
            "cancel_transfer",
            "Cancel a synthetic pending transfer by recipient.",
            {"recipient": {"type": ["string", "null"]}},
        ),
    ]


def generate_records(
    *,
    pilot_count: int = 1200,
    split_seed: int = 7303,
    synthetic_bank_path: Path = DEFAULT_SYNTHETIC_BANK_PATH,
) -> list[dict[str, Any]]:
    if pilot_count < len(_base_scenarios()):
        raise BankingToolSftDataError(
            f"pilot_count must be at least {len(_base_scenarios())} to cover required cases"
        )
    bank_path = _resolve_bank_path(synthetic_bank_path)
    scenarios = _expand_scenarios(pilot_count)
    records = [_scenario_to_record(scenario, bank_path=bank_path) for scenario in scenarios]
    _assign_splits(records, split_seed=split_seed)
    validate_records(records, synthetic_bank_path=bank_path)
    return records


def prepare(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    pilot_count: int = 1200,
    split_seed: int = 7303,
    synthetic_bank_path: Path = DEFAULT_SYNTHETIC_BANK_PATH,
    export_teacher_requests: Path | None = None,
    teacher_responses: Path | None = None,
    teacher_model: str | None = None,
    teacher_prompt_hash: str | None = None,
) -> dict[str, Any]:
    records = generate_records(
        pilot_count=pilot_count,
        split_seed=split_seed,
        synthetic_bank_path=synthetic_bank_path,
    )
    if export_teacher_requests is not None:
        export_teacher_realization_requests(records, export_teacher_requests)
    if teacher_responses is not None:
        if not teacher_model or not teacher_prompt_hash:
            raise BankingToolSftDataError(
                "--teacher-model and --teacher-prompt-hash are required with --teacher-responses"
            )
        records = import_teacher_realizations(
            records,
            teacher_responses,
            teacher_model=teacher_model,
            teacher_prompt_hash=teacher_prompt_hash,
        )
    split_rows = {
        split: [record for record in records if record["metadata"]["split"] == split]
        for split in SPLITS
    }
    report = _build_report(records, split_rows, split_seed=split_seed)
    _validate_report(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    entries = _write_split_files(output_dir, split_rows)
    manifest = {
        "format_version": 1,
        "name": "retail-bank-servicing-v3-tool-sft",
        "created_at": CREATED_AT,
        "contract": BANKING_TOOL_SFT_MANIFEST_CONTRACT,
        "schema_version": BANKING_TOOL_SFT_CONTRACT,
        "tool_manifest_hash": _tool_manifest_hash(),
        "tool_sft": entries,
        "source_roles": {
            "self-authored-synthetic": {
                "role": "tool-use-sft",
                "license": "MIT",
                "trainable": True,
            },
            "PolyAI/banking77": {
                "role": "classifier-eval-only",
                "trainable": False,
            },
            "bitext/Bitext-retail-banking-llm-chatbot-training-dataset": {
                "role": "quarantine-or-rewrite-only",
                "trainable": False,
            },
        },
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    _atomic_write_json(output_dir / "preparation-report.json", report)
    data_card = _render_data_card(report)
    (output_dir / "README.md").write_text(data_card, encoding="utf-8")
    (output_dir / "DATA_CARD.md").write_text(data_card, encoding="utf-8")
    return report


def export_teacher_realization_requests(records: Iterable[dict[str, Any]], path: Path) -> None:
    """Write teacher prompts while keeping semantic tool data separately hashable."""

    rows = []
    for record in records:
        rows.append(
            {
                "record_id": _required_str(record, "record_id"),
                "immutable_hash": _immutable_record_hash(record),
                "user_content": _last_user_message(record)["content"],
                "final_response": _final_assistant_message(record)["content"],
                "allowed_edits": ["user_content", "final_response"],
                "immutable_fields": [
                    "assistant tool_calls",
                    "tool messages",
                    "expected ordered_calls",
                    "expected final_state_hash",
                    "expected grounding_facts",
                    "split_keys",
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def import_teacher_realizations(
    records: Iterable[dict[str, Any]],
    path: Path,
    *,
    teacher_model: str,
    teacher_prompt_hash: str,
) -> list[dict[str, Any]]:
    """Apply teacher wording only after proving calls, results, and facts are unchanged."""

    rows = {_required_str(row, "record_id"): row for row in _read_jsonl(path)}
    realized = json.loads(json.dumps(list(records), sort_keys=True))
    for record in realized:
        record_id = _required_str(record, "record_id")
        row = rows.get(record_id)
        if row is None:
            continue
        before_hash = _immutable_record_hash(record)
        if row.get("immutable_hash") != before_hash:
            raise BankingToolSftDataError(f"{record_id} teacher request hash mismatch")
        user_content = row.get("user_content")
        final_response = row.get("final_response")
        if not isinstance(user_content, str) or not user_content.strip():
            raise BankingToolSftDataError(f"{record_id} teacher user_content must be text")
        if not isinstance(final_response, str) or not final_response.strip():
            raise BankingToolSftDataError(f"{record_id} teacher final_response must be text")
        _last_user_message(record)["content"] = user_content.strip()
        _final_assistant_message(record)["content"] = final_response.strip()
        if _immutable_record_hash(record) != before_hash:
            raise BankingToolSftDataError(f"{record_id} teacher changed immutable semantics")
        record["provenance"]["teacher_model"] = teacher_model
        record["provenance"]["teacher_prompt_hash"] = teacher_prompt_hash
        record["validation"]["teacher_realization_hash"] = (
            f"sha256:{hashlib.sha256(canonical_json_bytes(row)).hexdigest()}"
        )
    validate_records(realized)
    return realized


def validate_records(
    records: Iterable[dict[str, Any]],
    *,
    synthetic_bank_path: Path = DEFAULT_SYNTHETIC_BANK_PATH,
) -> None:
    tool_names = set(ALLOWED_ARGS)
    seen_ids: set[str] = set()
    normalized_users: set[str] = set()
    for record in records:
        record_id = _required_str(record, "record_id")
        if record_id in seen_ids:
            raise BankingToolSftDataError(f"duplicate record_id: {record_id}")
        seen_ids.add(record_id)
        if record.get("schema_version") != BANKING_TOOL_SFT_CONTRACT:
            raise BankingToolSftDataError(f"{record_id} has unexpected schema_version")
        if record.get("provenance", {}).get("source") != "self-authored-synthetic":
            raise BankingToolSftDataError(f"{record_id} has unsupported provenance")
        split_keys = record.get("split_keys")
        if not isinstance(split_keys, dict) or set(split_keys) != {
            "scenario_family",
            "state_seed",
            "customer_id",
            "template_id",
            "realization_seed",
        }:
            raise BankingToolSftDataError(f"{record_id} has invalid split_keys")
        ordered_calls = record.get("expected", {}).get("ordered_calls")
        if not isinstance(ordered_calls, list):
            raise BankingToolSftDataError(f"{record_id} missing ordered_calls")
        expected_calls = record.get("expected", {}).get("tool_calls")
        if not isinstance(expected_calls, list):
            raise BankingToolSftDataError(f"{record_id} missing expected tool_calls")
        user_key = normalized_user_text(_last_user_message(record)["content"])
        if user_key in normalized_users:
            raise BankingToolSftDataError(f"{record_id} duplicates normalized user text")
        normalized_users.add(user_key)
        final_response = _final_assistant_message(record).get("content")
        if not isinstance(final_response, str) or len(
            normalized_user_text(final_response).split()
        ) < 7:
            raise BankingToolSftDataError(
                f"{record_id} final assistant response is missing semantic content"
            )
        if _final_assistant_message(record).get("loss") is not True:
            raise BankingToolSftDataError(
                f"{record_id} final assistant response must be trainable"
            )
        normalized_final = normalized_user_text(final_response)
        response_path = record.get("expected", {}).get("path")
        required_path_markers = {
            "clarification": ("last four digits",),
            "ood": ("retail banking",),
            "hard_negative": ("account numbers", "customer ids"),
        }
        if response_path == "no_tool_banking_faq":
            template_id = str(split_keys["template_id"])
            required_path_markers["no_tool_banking_faq"] = FAQ_REQUIRED_MARKERS.get(
                template_id,
                (),
            )
        missing_markers = [
            marker
            for marker in required_path_markers.get(str(response_path), ())
            if normalized_user_text(marker) not in normalized_final
        ]
        if missing_markers:
            raise BankingToolSftDataError(
                f"{record_id} final assistant response is missing path markers: "
                f"{missing_markers}"
            )
        tool_call_ids: list[str] = []
        context_tool_call_ids: list[str] = []
        all_tool_call_ids: set[str] = set()
        canonical_calls: list[dict[str, Any]] = []
        tool_result_ids: list[str] = []
        pending_tool_call_id: str | None = None
        pending_tool_call_is_target = False
        for message in record.get("messages", []):
            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                if pending_tool_call_id is not None:
                    raise BankingToolSftDataError(f"{record_id} tool result correlation mismatch")
                if len(message["tool_calls"]) != 1:
                    raise BankingToolSftDataError(
                        f"{record_id} tool-call assistant must contain exactly one call"
                    )
                if message.get("content") is not None:
                    raise BankingToolSftDataError(f"{record_id} tool-call assistant has content")
                assistant_loss = message.get("loss")
                if assistant_loss not in {True, False}:
                    raise BankingToolSftDataError(
                        f"{record_id} assistant call has invalid loss label"
                    )
                for call in message["tool_calls"]:
                    call_id = _required_str(call, "id")
                    if call_id in all_tool_call_ids:
                        raise BankingToolSftDataError(
                            f"{record_id} has duplicate tool call id"
                        )
                    all_tool_call_ids.add(call_id)
                    index = call.get("index")
                    if index != 0:
                        raise BankingToolSftDataError(
                            f"{record_id} tool call index must restart at zero per "
                            "assistant message"
                        )
                    if assistant_loss is True:
                        global_call_index = len(tool_call_ids)
                        if call_id != f"call_{record_id}_{global_call_index}":
                            raise BankingToolSftDataError(
                                f"{record_id} has unstable tool call id"
                            )
                    else:
                        context_call_index = len(context_tool_call_ids)
                        if call_id != f"context_{record_id}_{context_call_index}":
                            raise BankingToolSftDataError(
                                f"{record_id} has unstable context tool call id"
                            )
                    function = call.get("function", {})
                    name = function.get("name")
                    arguments = function.get("arguments")
                    if name not in tool_names:
                        raise BankingToolSftDataError(f"{record_id} uses unknown tool {name!r}")
                    if not isinstance(arguments, dict):
                        raise BankingToolSftDataError(f"{record_id} tool arguments must be object")
                    extras = set(arguments) - ALLOWED_ARGS[str(name)]
                    if extras:
                        raise BankingToolSftDataError(
                            f"{record_id} unsupported arguments for {name}: {sorted(extras)}"
                        )
                    pending_tool_call_id = call_id
                    pending_tool_call_is_target = assistant_loss is True
                    if assistant_loss is True:
                        tool_call_ids.append(call_id)
                        canonical_calls.append(
                            {
                                "name": str(name),
                                "arguments": dict(arguments),
                            }
                        )
                    else:
                        context_tool_call_ids.append(call_id)
            elif role == "tool":
                if message.get("loss") is not False:
                    raise BankingToolSftDataError(f"{record_id} tool result is labeled")
                content = message.get("content")
                if not isinstance(content, dict):
                    raise BankingToolSftDataError(f"{record_id} tool content must be object")
                if content.get("ok") is True and set(content) != {"ok", "result"}:
                    raise BankingToolSftDataError(f"{record_id} success envelope is invalid")
                if content.get("ok") is False and set(content) != {"ok", "error"}:
                    raise BankingToolSftDataError(f"{record_id} error envelope is invalid")
                if content.get("ok") not in {True, False}:
                    raise BankingToolSftDataError(f"{record_id} tool envelope missing ok")
                tool_call_id = _required_str(message, "tool_call_id")
                if pending_tool_call_id != tool_call_id:
                    raise BankingToolSftDataError(f"{record_id} tool result correlation mismatch")
                if pending_tool_call_is_target:
                    tool_result_ids.append(tool_call_id)
                pending_tool_call_id = None
                pending_tool_call_is_target = False
            elif role == "assistant":
                if pending_tool_call_id is not None:
                    raise BankingToolSftDataError(f"{record_id} tool result correlation mismatch")
                if message.get("loss") not in {True, False}:
                    raise BankingToolSftDataError(
                        f"{record_id} assistant message has invalid loss label"
                    )
            elif role in {"system", "user"}:
                if pending_tool_call_id is not None:
                    raise BankingToolSftDataError(f"{record_id} tool result correlation mismatch")
                if message.get("loss") is not False:
                    raise BankingToolSftDataError(f"{record_id} context message is labeled")
            else:
                raise BankingToolSftDataError(f"{record_id} has invalid role {role!r}")
        if pending_tool_call_id is not None:
            raise BankingToolSftDataError(f"{record_id} tool result correlation mismatch")
        if tool_call_ids != ordered_calls:
            raise BankingToolSftDataError(f"{record_id} ordered_calls mismatch")
        if canonical_calls != expected_calls:
            raise BankingToolSftDataError(f"{record_id} expected tool_calls mismatch")
        if tool_result_ids != tool_call_ids:
            raise BankingToolSftDataError(f"{record_id} tool result correlation mismatch")
        if bool(tool_call_ids) != bool(record.get("expected", {}).get("requires_tool")):
            raise BankingToolSftDataError(f"{record_id} requires_tool mismatch")
        if record.get("validation", {}).get("tool_manifest_hash") != _tool_manifest_hash():
            raise BankingToolSftDataError(f"{record_id} manifest hash mismatch")
    _ = synthetic_bank_path


def validate_banking_tool_sft_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("contract") != BANKING_TOOL_SFT_MANIFEST_CONTRACT:
        raise BankingToolSftDataError("not a banking-v3 tool SFT manifest")
    if manifest.get("schema_version") != BANKING_TOOL_SFT_CONTRACT:
        raise BankingToolSftDataError("manifest schema_version mismatch")
    base = path.parent
    all_records: list[dict[str, Any]] = []
    for entry in manifest.get("tool_sft", []):
        declared_path = Path(entry["path"])
        if declared_path.is_absolute():
            raise BankingToolSftDataError(f"{entry['name']} path must be manifest-relative")
        rows = _read_jsonl(base / declared_path)
        if len(rows) != int(entry["record_count"]):
            raise BankingToolSftDataError(f"{entry['name']} record_count mismatch")
        all_records.extend(rows)
    validate_records(all_records)
    return manifest


def normalized_user_text(text: str) -> str:
    return NORMALIZED_TEXT_RE.sub(" ", text.lower()).strip()


def _tool(name: str, description: str, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "additionalProperties": False,
            },
        },
    }


def _base_scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            "accounts_read",
            "read_accounts",
            "accounts-balance-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "What accounts do I have and what are their balances?",
            "Everyday Checking ending in 1042 has USD 3,245.67 available and "
            "USD 3,300.12 current. Goal Saver ending in 8831 has USD 12,500.00 "
            "available and current.",
            "tool_success",
            (ToolPlan("list_accounts", {}),),
            grounding_facts=(
                "accounts.count=2",
                "account.last4=1042",
                "account.last4=8831",
                "account.balance=3,245.67",
                "account.balance=12,500.00",
            ),
        ),
        Scenario(
            "cards_read",
            "read_cards",
            "cards-status-v1",
            "realization-000",
            "maya.demo",
            "synthetic-customer-maya",
            "state-maya-001",
            "Is my debit card active?",
            "Your Travel Visa Debit card ending in 7319 is active and is not added to a wallet.",
            "tool_success",
            (ToolPlan("list_cards", {}),),
            grounding_facts=("card.last4=7319", "card.status=active"),
        ),
        Scenario(
            "cases_read",
            "read_service_cases",
            "cases-status-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Do I have any service cases?",
            "You have a closed case for confirming a mailing address update.",
            "tool_success",
            (ToolPlan("list_service_cases", {}),),
            grounding_facts=("case.status=closed", "case.case_type=address_update"),
        ),
        Scenario(
            "transactions_read",
            "read_transactions",
            "transactions-recent-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Show my three most recent transactions.",
            "Your three most recent transactions include North Harbor Market, "
            "CloudStream, and Harbor Labs Payroll.",
            "tool_success",
            (ToolPlan("list_transactions", {"limit": 3}),),
            grounding_facts=("transaction.description=North Harbor Market", "transactions.limit=3"),
        ),
        Scenario(
            "transfers_read",
            "read_transfers",
            "transfers-status-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Which transfers are on my account?",
            "You have a pending transfer to River Consulting and a completed transfer "
            "to Jamie Lee.",
            "tool_success",
            (ToolPlan("list_transfers", {}),),
            grounding_facts=("transfer.recipient=River Consulting", "transfer.status=pending"),
        ),
        Scenario(
            "freeze_card_success",
            "card_freeze",
            "freeze-explicit-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Freeze my debit card ending in 4821.",
            "Your debit card ending in 4821 is now frozen.",
            "tool_success",
            (ToolPlan("freeze_card", {"last4": "4821"}),),
            grounding_facts=("card.last4=4821", "card.status=frozen"),
        ),
        Scenario(
            "replace_card_success",
            "card_replace",
            "replace-explicit-v1",
            "realization-000",
            "maya.demo",
            "synthetic-customer-maya",
            "state-maya-001",
            "Please replace the card ending 7319.",
            "Replacement is pending for your card ending in 7319.",
            "tool_success",
            (ToolPlan("replace_card", {"last4": "7319"}),),
            grounding_facts=("card.last4=7319", "card.status=replacement_pending"),
        ),
        Scenario(
            "dispute_transaction_success",
            "transaction_dispute",
            "dispute-merchant-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "I need to dispute the North Harbor Market debit.",
            "I opened a dispute for the North Harbor Market transaction.",
            "tool_success",
            (ToolPlan("dispute_transaction", {"description": "North Harbor Market"}),),
            grounding_facts=(
                "transaction.description=North Harbor Market",
                "transaction.disputed=true",
            ),
        ),
        Scenario(
            "cancel_transfer_success",
            "transfer_cancel",
            "cancel-recipient-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Cancel my pending transfer to River Consulting.",
            "The pending transfer to River Consulting is now cancelled.",
            "tool_success",
            (ToolPlan("cancel_transfer", {"recipient": "River Consulting"}),),
            grounding_facts=("transfer.recipient=River Consulting", "transfer.status=cancelled"),
        ),
        Scenario(
            "multi_tool_freeze",
            "multi_tool_card_action",
            "list-then-freeze-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Check my cards, then freeze the active debit card.",
            "I found your active debit card ending in 4821 and froze it.",
            "multi_turn",
            (ToolPlan("list_cards", {}), ToolPlan("freeze_card", {"last4": "4821"})),
            grounding_facts=("card.last4=4821", "card.status=frozen"),
        ),
        Scenario(
            "emergency_card_freeze",
            "emergency_card_freeze",
            "stolen-list-then-freeze-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "My card was stolen. Freeze it.",
            "I found your active debit card ending in 4821 and froze it.",
            "multi_turn",
            (ToolPlan("list_cards", {}), ToolPlan("freeze_card", {"last4": "4821"})),
            grounding_facts=("card.last4=4821", "card.status=frozen"),
        ),
        Scenario(
            "action_summary_followup",
            "action_summary_followup",
            "summarize-card-freeze-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "What did you just do?",
            "I found the active debit card ending in 4821 and froze it.",
            "multi_turn",
            grounding_facts=("card.last4=4821", "card.status=frozen"),
        ),
        Scenario(
            "no_action_followup",
            "no_action_followup",
            "summarize-clarification-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Did you replace it already?",
            "No. I only asked which card you want replaced; no card change was made.",
            "multi_turn",
        ),
        Scenario(
            "clarify_card_missing",
            "clarification_card",
            "clarify-card-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Replace my card.",
            "Which card should I replace? Please share the last four digits shown in "
            "the app.",
            "clarification",
            grounding_facts=("missing_field=last4",),
        ),
        Scenario(
            "general_banking_faq",
            "no_tool_banking_faq",
            "faq-overdraft-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "How do overdraft fees usually work?",
            "Overdraft fees can apply when a transaction exceeds your available "
            "balance; review your account disclosures or settings for exact terms.",
            "no_tool_banking_faq",
            grounding_facts=("faq=overdraft",),
        ),
        Scenario(
            "mortgage_opening_faq",
            "faq_mortgage",
            "faq-mortgage-opening-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Can you help me open a mortgage account?",
            "I can explain mortgage basics and typical application steps, but this "
            "demo cannot open or approve a mortgage. A real lender would review an "
            "application and disclose its terms through an approved channel.",
            "no_tool_banking_faq",
            grounding_facts=("faq=mortgage",),
        ),
        Scenario(
            "mortgage_age_faq",
            "faq_mortgage_age",
            "faq-mortgage-age-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "How old do I need to be to apply for a mortgage?",
            "Mortgage applicants generally must be legal adults—typically at least "
            "18 in the United States. Requirements vary by lender and jurisdiction, "
            "and this demo cannot determine your eligibility.",
            "no_tool_banking_faq",
            grounding_facts=("faq=mortgage", "mortgage.minimum_age=18"),
        ),
        Scenario(
            "deposit_account_opening_faq",
            "faq_deposit_opening",
            "faq-deposit-opening-v1",
            "realization-000",
            "maya.demo",
            "synthetic-customer-maya",
            "state-maya-001",
            "How would I open a new savings account?",
            "This demo cannot open a real account. Banks typically ask you to choose "
            "an account type, review disclosures, verify identity through an approved "
            "channel, and provide opening funds when required.",
            "no_tool_banking_faq",
            grounding_facts=("faq=account",),
        ),
        Scenario(
            "savings_interest_faq",
            "faq_savings_interest",
            "faq-savings-interest-v1",
            "realization-000",
            "maya.demo",
            "synthetic-customer-maya",
            "state-maya-001",
            "How does interest on a savings account work?",
            "Savings interest is generally calculated from the balance and stated "
            "annual percentage yield, then credited on the bank’s schedule. Exact "
            "rates and compounding rules depend on the account disclosures.",
            "no_tool_banking_faq",
            grounding_facts=("faq=savings_interest",),
        ),
        Scenario(
            "small_talk_greeting",
            "small_talk_greeting",
            "small-talk-greeting-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Yo, sup?",
            "Hi! I’m ready to help with your banking questions or the synthetic "
            "accounts in this demo.",
            "conversation",
        ),
        Scenario(
            "small_talk_checkin",
            "small_talk_checkin",
            "small-talk-checkin-v1",
            "realization-000",
            "maya.demo",
            "synthetic-customer-maya",
            "state-maya-001",
            "How are you doing?",
            "I’m ready to help. What banking question or account task would you like "
            "to work on?",
            "conversation",
        ),
        Scenario(
            "conversational_thanks",
            "conversational_thanks",
            "conversation-thanks-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Thanks for the help.",
            "You’re welcome. I can help with another banking question whenever you’re "
            "ready.",
            "conversation",
        ),
        Scenario(
            "ood_weather",
            "ood",
            "ood-weather-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Will it rain in Seattle tomorrow?",
            NO_TOOL_OOD_RESPONSE,
            "ood",
            grounding_facts=("domain=out_of_domain",),
        ),
        Scenario(
            "hard_negative_account_number",
            "hard_negative_private_id",
            "private-account-number-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Tell me my full account number and customer ID.",
            "I cannot provide full account numbers or customer IDs in chat. I can help "
            "with balances, card status, transfers, and other supported banking tasks.",
            "hard_negative",
            grounding_facts=("private_data_refused=true",),
        ),
        Scenario(
            "transfer_cancel_completed_error",
            "backend_error",
            "cancel-completed-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "Cancel my transfer to Jamie Lee.",
            "I could not cancel a pending transfer to Jamie Lee because no matching "
            "pending transfer was found.",
            "tool_error",
            (ToolPlan("cancel_transfer", {"recipient": "Jamie Lee"}),),
            grounding_facts=("error.code=backend_error",),
        ),
        Scenario(
            "dispute_missing_error",
            "backend_error",
            "dispute-missing-v1",
            "realization-000",
            "maya.demo",
            "synthetic-customer-maya",
            "state-maya-001",
            "Dispute the Demo Bookstore charge.",
            "I could not open that dispute because no matching eligible transaction was found.",
            "tool_error",
            (ToolPlan("dispute_transaction", {"description": "Demo Bookstore"}),),
            grounding_facts=("error.code=backend_error",),
        ),
        Scenario(
            "two_turn_dispute",
            "multi_turn_dispute",
            "clarify-then-dispute-v1",
            "realization-000",
            "alex.demo",
            "synthetic-customer-alex",
            "state-alex-001",
            "It was North Harbor Market.",
            "I opened a dispute for the North Harbor Market transaction.",
            "multi_turn",
            (ToolPlan("dispute_transaction", {"description": "North Harbor Market"}),),
            pre_messages=(
                _message("user", "I need to dispute a charge.", loss=False),
                _message("assistant", "Which merchant or transaction should I dispute?", loss=True),
            ),
            grounding_facts=(
                "transaction.description=North Harbor Market",
                "transaction.disputed=true",
            ),
        ),
    )


def _expand_scenarios(pilot_count: int) -> list[Scenario]:
    base = _base_scenarios()
    scenarios = []
    realization_counts = {scenario.scenario_id: 0 for scenario in base}
    for index in range(pilot_count):
        template = base[index % len(base)]
        occurrence = realization_counts[template.scenario_id]
        realization_counts[template.scenario_id] += 1
        scenarios.append(_materialize_scenario(template, occurrence))
    return scenarios


def _materialize_scenario(template: Scenario, occurrence: int) -> Scenario:
    semantic_seed = occurrence // 3
    realization_index = occurrence % 3
    slot = _state_slots(template, semantic_seed)
    scenario_id = (
        template.scenario_id
        if occurrence == 0
        else f"{template.scenario_id}_state_{semantic_seed:04d}_realization_{realization_index}"
    )
    material = Scenario(
        scenario_id=scenario_id,
        scenario_family=template.scenario_family,
        template_id=template.template_id,
        realization_seed=f"realization-{realization_index:03d}",
        customer_login=str(slot["login"]),
        customer_id=str(slot["customer_id"]),
        state_seed=str(slot["state_seed"]),
        user="",
        final_response="",
        path=template.path,
        tool_plan=_materialized_tool_plan(template, slot),
        pre_messages=_materialized_pre_messages(template, slot, scenario_id),
        grounding_facts=_materialized_grounding_facts(template, slot),
        bank_payload=_synthetic_bank_payload(slot),
    )
    return Scenario(
        scenario_id=material.scenario_id,
        scenario_family=material.scenario_family,
        template_id=material.template_id,
        realization_seed=material.realization_seed,
        customer_login=material.customer_login,
        customer_id=material.customer_id,
        state_seed=material.state_seed,
        user=_realize_user(material, occurrence),
        final_response=_materialized_final_response(template, slot, occurrence),
        path=material.path,
        tool_plan=material.tool_plan,
        pre_messages=material.pre_messages,
        grounding_facts=material.grounding_facts,
        bank_payload=material.bank_payload,
    )


def _state_slots(template: Scenario, semantic_seed: int) -> dict[str, Any]:
    seed = _stable_int(f"{template.scenario_id}:{semantic_seed}")
    checking_last4 = _last4(seed, 11)
    savings_last4 = _last4(seed, 23)
    card_last4 = _last4(seed, 37)
    merchant = f"{_pick(MERCHANT_PREFIXES, seed)} {_pick(MERCHANT_TYPES, seed // 7)}"
    alternate_merchant = (
        f"{_pick(MERCHANT_PREFIXES, seed // 11 + 3)} "
        f"{_pick(MERCHANT_TYPES, seed // 13 + 4)}"
    )
    missing_merchant = (
        f"{_pick(MERCHANT_PREFIXES, seed // 17 + 6)} "
        f"{_pick(MERCHANT_TYPES, seed // 19 + 7)}"
    )
    pending_recipient = (
        f"{_pick(RECIPIENT_PREFIXES, seed)} {_pick(RECIPIENT_TYPES, seed // 5)}"
    )
    completed_recipient = (
        f"{_pick(RECIPIENT_PREFIXES, seed // 7 + 4)} "
        f"{_pick(RECIPIENT_TYPES, seed // 11 + 3)}"
    )
    checking_name = _pick(ACCOUNT_LABELS, seed)
    savings_name = _pick(SAVINGS_LABELS, seed // 3)
    card_name = _pick(CARD_LABELS, seed // 5)
    checking_available = 75_000 + (seed % 420_000)
    savings_available = 250_000 + ((seed // 3) % 1_900_000)
    transaction_amount = -(1_000 + (seed % 32_000))
    transfer_amount = 2_500 + ((seed // 5) % 175_000)
    customer_id = f"cust_tool_{template.scenario_id}_{semantic_seed:04d}"
    login = f"{template.scenario_id}.{semantic_seed:04d}.demo"
    return {
        "semantic_seed": semantic_seed,
        "state_seed": f"state-{template.scenario_id}-{semantic_seed:04d}",
        "customer_id": customer_id,
        "login": login,
        "display_name": (
            f"{_pick(FICTIONAL_FIRST_NAMES, seed)} "
            f"{_pick(FICTIONAL_LAST_NAMES, seed // 9)}"
        ),
        "city": _pick(("North Harbor", "Pine Ridge", "Lakeview", "Cedar Point"), seed),
        "checking_name": checking_name,
        "checking_last4": checking_last4,
        "checking_available": checking_available,
        "checking_current": checking_available + 4_500,
        "savings_name": savings_name,
        "savings_last4": savings_last4,
        "savings_available": savings_available,
        "card_name": card_name,
        "card_last4": card_last4,
        "merchant": merchant,
        "alternate_merchant": alternate_merchant,
        "missing_merchant": missing_merchant,
        "transaction_amount": transaction_amount,
        "pending_recipient": pending_recipient,
        "completed_recipient": completed_recipient,
        "transfer_amount": transfer_amount,
    }


def _materialized_tool_plan(template: Scenario, slot: dict[str, Any]) -> tuple[ToolPlan, ...]:
    plans = []
    for call in template.tool_plan:
        arguments = dict(call.arguments)
        if call.name in {"freeze_card", "replace_card"}:
            arguments["last4"] = slot["card_last4"]
        elif call.name == "dispute_transaction":
            arguments["description"] = (
                slot["missing_merchant"]
                if template.template_id == "dispute-missing-v1"
                else slot["merchant"]
            )
        elif call.name == "cancel_transfer":
            arguments["recipient"] = (
                slot["completed_recipient"]
                if template.template_id == "cancel-completed-v1"
                else slot["pending_recipient"]
            )
        plans.append(ToolPlan(call.name, arguments))
    return tuple(plans)


def _materialized_pre_messages(
    template: Scenario,
    slot: dict[str, Any],
    record_id: str,
) -> tuple[dict[str, Any], ...]:
    family = template.scenario_family
    if family == "multi_turn_dispute":
        return (
            _message("user", "I need to dispute a debit card charge.", loss=False),
            _message("assistant", "Which merchant or transaction should I dispute?", loss=True),
        )
    if family == "action_summary_followup":
        card_before = _context_card(slot, status="active")
        card_after = _context_card(slot, status="frozen")
        list_call_id = f"context_{record_id}_0"
        freeze_call_id = f"context_{record_id}_1"
        return (
            _message(
                "user",
                "My wallet was stolen. Find my active debit card and freeze it.",
                loss=False,
            ),
            {
                "role": "assistant",
                "content": None,
                "loss": False,
                "tool_calls": [
                    {
                        "id": list_call_id,
                        "index": 0,
                        "type": "function",
                        "function": {"name": "list_cards", "arguments": {}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": list_call_id,
                "name": "list_cards",
                "content": {"ok": True, "result": {"cards": [card_before]}},
                "loss": False,
            },
            {
                "role": "assistant",
                "content": None,
                "loss": False,
                "tool_calls": [
                    {
                        "id": freeze_call_id,
                        "index": 0,
                        "type": "function",
                        "function": {
                            "name": "freeze_card",
                            "arguments": {"last4": slot["card_last4"]},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": freeze_call_id,
                "name": "freeze_card",
                "content": {
                    "ok": True,
                    "result": {"card": card_after, "simulated": True},
                },
                "loss": False,
            },
            _message(
                "assistant",
                f"I found your active debit card ending in {slot['card_last4']} and "
                "froze it.",
                loss=False,
            ),
        )
    if family == "no_action_followup":
        return (
            _message("user", "Replace my card.", loss=False),
            _message(
                "assistant",
                "Which card should I replace? Please share the last four digits shown "
                "in the app.",
                loss=False,
            ),
        )
    return template.pre_messages


def _context_card(slot: dict[str, Any], *, status: str) -> dict[str, Any]:
    customer_id = str(slot["customer_id"])
    return {
        "account_id": f"acct_{customer_id}_checking",
        "card_id": f"card_{customer_id}_debit",
        "customer_id": customer_id,
        "last4": slot["card_last4"],
        "name": slot["card_name"],
        "status": status,
        "wallet_status": "added",
    }


def _materialized_grounding_facts(
    template: Scenario, slot: dict[str, Any]
) -> tuple[str, ...]:
    family = template.scenario_family
    if family == "read_accounts":
        return (
            "accounts.count=2",
            f"account.last4={slot['checking_last4']}",
            f"account.last4={slot['savings_last4']}",
            f"account.balance={_format_usd(slot['checking_available'])}",
            f"account.balance={_format_usd(slot['savings_available'])}",
        )
    if family == "read_cards":
        return (f"card.last4={slot['card_last4']}", "card.status=active")
    if family == "read_service_cases":
        return ("case.status=closed", "case.case_type=address_update")
    if family == "read_transactions":
        return (f"transaction.description={slot['merchant']}", "transactions.limit=3")
    if family == "read_transfers":
        return (f"transfer.recipient={slot['pending_recipient']}", "transfer.status=pending")
    if family in {
        "card_freeze",
        "multi_tool_card_action",
        "emergency_card_freeze",
        "action_summary_followup",
    }:
        return (f"card.last4={slot['card_last4']}", "card.status=frozen")
    if family == "card_replace":
        return (f"card.last4={slot['card_last4']}", "card.status=replacement_pending")
    if family in {"transaction_dispute", "multi_turn_dispute"}:
        return (f"transaction.description={slot['merchant']}", "transaction.disputed=true")
    if family == "transfer_cancel":
        return (f"transfer.recipient={slot['pending_recipient']}", "transfer.status=cancelled")
    return template.grounding_facts


def _materialized_final_response(
    template: Scenario, slot: dict[str, Any], occurrence: int
) -> str:
    family = template.scenario_family
    if family == "read_accounts":
        final = (
            f"{slot['checking_name']} ending in {slot['checking_last4']} has USD "
            f"{_format_usd(slot['checking_available'])} available and USD "
            f"{_format_usd(slot['checking_current'])} current. "
            f"{slot['savings_name']} ending in {slot['savings_last4']} has USD "
            f"{_format_usd(slot['savings_available'])} available and current."
        )
    elif family == "read_cards":
        final = (
            f"Your {slot['card_name']} ending in {slot['card_last4']} is active and "
            "available for use."
        )
    elif family == "read_service_cases":
        final = "You have a closed service case confirming a mailing address update."
    elif family == "read_transactions":
        final = (
            f"Your recent transactions include {slot['merchant']}, "
            f"{slot['alternate_merchant']}, and payroll."
        )
    elif family == "read_transfers":
        final = (
            f"You have a pending transfer to {slot['pending_recipient']} and a completed "
            f"transfer to {slot['completed_recipient']}."
        )
    elif family == "card_freeze":
        final = f"Your debit card ending in {slot['card_last4']} is now frozen."
    elif family == "card_replace":
        final = f"Replacement is pending for your card ending in {slot['card_last4']}."
    elif family == "transaction_dispute":
        final = f"I opened a dispute for the {slot['merchant']} transaction."
    elif family == "transfer_cancel":
        final = f"The pending transfer to {slot['pending_recipient']} is now cancelled."
    elif family in {"multi_tool_card_action", "emergency_card_freeze"}:
        final = (
            f"I found your active debit card ending in {slot['card_last4']} and froze it."
        )
    elif family == "action_summary_followup":
        final = (
            f"I found the active debit card ending in {slot['card_last4']} and froze it."
        )
    elif family == "no_action_followup":
        final = (
            "No. I only asked which card you want replaced; no card change was made."
        )
    elif family == "backend_error" and template.template_id == "cancel-completed-v1":
        final = (
            f"I could not cancel a pending transfer to {slot['completed_recipient']} "
            "because no matching pending transfer was found."
        )
    elif family == "backend_error":
        final = (
            f"I could not open that dispute because no eligible transaction for "
            f"{slot['missing_merchant']} was found."
        )
    elif family == "multi_turn_dispute":
        final = f"I opened a dispute for the {slot['merchant']} transaction."
    else:
        final = template.final_response
    return _realize_final(Scenario(**{**template.__dict__, "final_response": final}), occurrence)


def _synthetic_bank_payload(slot: dict[str, Any]) -> dict[str, Any]:
    customer_id = str(slot["customer_id"])
    checking_id = f"acct_{customer_id}_checking"
    savings_id = f"acct_{customer_id}_savings"
    return {
        "contract": "synthetic-retail-bank-v1",
        "notice": "All people, identifiers, balances, and events are fictional test data.",
        "customers": [
            {
                "customer_id": customer_id,
                "login": slot["login"],
                "display_name": slot["display_name"],
                "segment": "Synthetic Banking",
                "city": slot["city"],
                "member_since": "2022-01-15",
                "accounts": [
                    {
                        "account_id": checking_id,
                        "name": slot["checking_name"],
                        "type": "checking",
                        "last4": slot["checking_last4"],
                        "currency": "USD",
                        "available_balance_cents": slot["checking_available"],
                        "current_balance_cents": slot["checking_current"],
                        "status": "active",
                    },
                    {
                        "account_id": savings_id,
                        "name": slot["savings_name"],
                        "type": "savings",
                        "last4": slot["savings_last4"],
                        "currency": "USD",
                        "available_balance_cents": slot["savings_available"],
                        "current_balance_cents": slot["savings_available"],
                        "status": "active",
                    },
                ],
                "cards": [
                    {
                        "card_id": f"card_{customer_id}_debit",
                        "account_id": checking_id,
                        "name": slot["card_name"],
                        "last4": slot["card_last4"],
                        "status": "active",
                        "wallet_status": "added",
                    }
                ],
                "transactions": [
                    {
                        "transaction_id": f"txn_{customer_id}_001",
                        "account_id": checking_id,
                        "posted_at": "2026-07-25T15:42:00Z",
                        "description": slot["merchant"],
                        "amount_cents": slot["transaction_amount"],
                        "currency": "USD",
                        "status": "posted",
                        "category": "shopping",
                        "disputed": False,
                    },
                    {
                        "transaction_id": f"txn_{customer_id}_002",
                        "account_id": checking_id,
                        "posted_at": "2026-07-24T13:08:00Z",
                        "description": slot["alternate_merchant"],
                        "amount_cents": -1499,
                        "currency": "USD",
                        "status": "posted",
                        "category": "subscriptions",
                        "disputed": False,
                    },
                    {
                        "transaction_id": f"txn_{customer_id}_003",
                        "account_id": checking_id,
                        "posted_at": "2026-07-23T09:00:00Z",
                        "description": "Fictional Payroll",
                        "amount_cents": 240000,
                        "currency": "USD",
                        "status": "posted",
                        "category": "income",
                        "disputed": False,
                    },
                ],
                "transfers": [
                    {
                        "transfer_id": f"trf_{customer_id}_100",
                        "from_account_id": checking_id,
                        "recipient": slot["pending_recipient"],
                        "amount_cents": slot["transfer_amount"],
                        "currency": "USD",
                        "created_at": "2026-07-25T16:10:00Z",
                        "status": "pending",
                        "reference": "Synthetic invoice",
                    },
                    {
                        "transfer_id": f"trf_{customer_id}_101",
                        "from_account_id": checking_id,
                        "recipient": slot["completed_recipient"],
                        "amount_cents": max(1000, int(slot["transfer_amount"]) // 2),
                        "currency": "USD",
                        "created_at": "2026-07-20T11:20:00Z",
                        "status": "completed",
                        "reference": "Synthetic completed payment",
                    },
                ],
                "service_cases": [
                    {
                        "case_id": f"svc_{customer_id}_001",
                        "case_type": "address_update",
                        "subject": "Confirm mailing address update",
                        "status": "closed",
                        "created_at": "2026-06-18T14:00:00Z",
                    }
                ],
            }
        ],
    }


def _scenario_to_record(scenario: Scenario, *, bank_path: Path) -> dict[str, Any]:
    record_id = scenario.scenario_id
    messages = [_message("system", SYSTEM_PROMPT, loss=False), *scenario.pre_messages]
    messages.append(_message("user", scenario.user, loss=False))
    tool_messages, final_state_hash, replay_hash = _replay_tool_plan(
        scenario,
        record_id=record_id,
        bank_path=bank_path,
    )
    ordered_calls = []
    expected_calls = []
    if scenario.tool_plan:
        for index, call in enumerate(scenario.tool_plan):
            call_id = f"call_{record_id}_{index}"
            ordered_calls.append(call_id)
            expected_calls.append(
                {
                    "name": call.name,
                    "arguments": dict(call.arguments),
                }
            )
            tool_call = {
                "id": call_id,
                "index": 0,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": dict(call.arguments),
                },
            }
            messages.append(
                {"role": "assistant", "content": None, "loss": True, "tool_calls": [tool_call]}
            )
            messages.append(tool_messages[index])
    messages.append(_message("assistant", scenario.final_response, loss=True))

    return {
        "schema_version": BANKING_TOOL_SFT_CONTRACT,
        "record_id": record_id,
        "messages": messages,
        "expected": {
            "requires_tool": bool(scenario.tool_plan),
            "ordered_calls": ordered_calls,
            "tool_calls": expected_calls,
            "final_state_hash": final_state_hash,
            "grounding_facts": list(scenario.grounding_facts),
            "path": scenario.path,
        },
        "split_keys": {
            "scenario_family": scenario.scenario_family,
            "state_seed": scenario.state_seed,
            "customer_id": scenario.customer_id,
            "template_id": scenario.template_id,
            "realization_seed": scenario.realization_seed,
        },
        "provenance": {
            "source": "self-authored-synthetic",
            "license": "MIT",
            "generator_version": GENERATOR_VERSION,
            "teacher_model": None,
            "teacher_prompt_hash": None,
        },
        "validation": {
            "tool_manifest_hash": _tool_manifest_hash(),
            "replay_hash": replay_hash,
            "accepted": True,
        },
        "metadata": {
            "record_type": "tool_use_sft",
            "trainable": True,
            "customer_login": scenario.customer_login,
            "scenario_family": scenario.scenario_family,
            "path": scenario.path,
            "split_group": _split_group(scenario),
        },
    }


def _replay_tool_plan(
    scenario: Scenario,
    *,
    record_id: str,
    bank_path: Path,
) -> tuple[list[dict[str, Any]], str | None, str]:
    bank = _load_bank_registry(bank_path, payload=scenario.bank_payload)
    session_hash = f"tool-sft:{record_id}"
    tool_messages = []
    replay_events: list[dict[str, Any]] = []
    for index, call in enumerate(scenario.tool_plan):
        call_id = f"call_{record_id}_{index}"
        try:
            result = bank.execute(
                scenario.customer_login,
                session_hash,
                call.name,
                dict(call.arguments),
            )
            envelope = {"ok": True, "result": result}
        except ValueError as exc:
            envelope = {
                "ok": False,
                "error": {
                    "code": "backend_error",
                    "message": _safe_error_message(str(exc)),
                },
            }
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": call.name,
                "content": envelope,
                "loss": False,
            }
        )
        replay_events.append({"tool_call_id": call_id, "name": call.name, "content": envelope})
    snapshot = _stable_snapshot(bank.snapshot(scenario.customer_login, session_hash))
    final_state_hash: str | None = (
        f"sha256:{hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest()}"
    )
    replay_hash = f"sha256:{hashlib.sha256(canonical_json_bytes(replay_events)).hexdigest()}"
    if not scenario.tool_plan:
        final_state_hash = None
    return tool_messages, final_state_hash, replay_hash


def _assign_splits(records: list[dict[str, Any]], *, split_seed: int) -> None:
    groups = sorted({record["metadata"]["split_group"] for record in records})
    if len(groups) < 3:
        raise BankingToolSftDataError("at least three split groups are required")
    assigned: dict[str, str] = {}
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{split_seed}\n{group}".encode()).hexdigest(),
    )
    validation_target = max(1, round(len(ordered) * 0.15))
    test_target = max(1, round(len(ordered) * 0.15))
    for group in ordered[:validation_target]:
        assigned[group] = "validation"
    for group in ordered[validation_target : validation_target + test_target]:
        assigned[group] = "test"
    for group in ordered[validation_target + test_target :]:
        assigned[group] = "train"
    for record in records:
        record["metadata"]["split"] = assigned[record["metadata"]["split_group"]]


def _build_report(
    records: list[dict[str, Any]],
    split_rows: dict[str, list[dict[str, Any]]],
    *,
    split_seed: int,
) -> dict[str, Any]:
    calls = [
        call["function"]["name"]
        for record in records
        for message in record["messages"]
        for call in message.get("tool_calls", [])
    ]
    paths = Counter(record["expected"]["path"] for record in records)
    canonical = sorted(records, key=lambda record: record["record_id"])
    return {
        "format_version": 1,
        "name": "retail-bank-servicing-v3-tool-sft",
        "created_at": CREATED_AT,
        "summary": {
            "total_records": len(records),
            "split_seed": split_seed,
            "corpus_fingerprint": hashlib.sha256(canonical_json_bytes(canonical)).hexdigest(),
        },
        "splits": {
            split: {
                "records": len(rows),
                "paths": dict(Counter(row["expected"]["path"] for row in rows)),
            }
            for split, rows in split_rows.items()
        },
        "checks": {
            "accepted_records": sum(1 for record in records if record["validation"]["accepted"]),
            "tool_names_covered": sorted(set(calls)),
            "required_tool_names": sorted(ALLOWED_ARGS),
            "success_tool_messages": sum(
                1
                for record in records
                for message in record["messages"]
                if message["role"] == "tool" and message["content"]["ok"] is True
            ),
            "error_tool_messages": sum(
                1
                for record in records
                for message in record["messages"]
                if message["role"] == "tool" and message["content"]["ok"] is False
            ),
            "paths": dict(paths),
            "banking77_generative_sft_rows": 0,
            "bitext_rows": 0,
        },
        "quarantine": {
            "bitext": {
                "trainable": False,
                "reason": "Bitext rows require audit/rewrite before any v3 tool-use inclusion.",
            }
        },
        "source_roles": {
            "Banking77/CLINC": "classifier-evaluation-only",
            "Bitext": "audit-quarantine-only",
            "self-authored-synthetic": "tool-use-sft",
        },
    }


def _validate_report(report: dict[str, Any]) -> None:
    checks = report["checks"]
    if checks["accepted_records"] != report["summary"]["total_records"]:
        raise BankingToolSftDataError("not all records were accepted")
    if checks["tool_names_covered"] != checks["required_tool_names"]:
        raise BankingToolSftDataError("not all banking tools are covered")
    if checks["success_tool_messages"] < 1 or checks["error_tool_messages"] < 1:
        raise BankingToolSftDataError("success and error envelopes are both required")
    required_paths = {
        "tool_success",
        "tool_error",
        "clarification",
        "no_tool_banking_faq",
        "ood",
        "hard_negative",
        "multi_turn",
    }
    missing = required_paths - set(checks["paths"])
    if missing:
        raise BankingToolSftDataError(f"missing required paths: {sorted(missing)}")
    for split, data in report["splits"].items():
        if data["records"] < 1:
            raise BankingToolSftDataError(f"{split} split is empty")


def _write_split_files(
    output_dir: Path, split_rows: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    entries = []
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        rows = sorted(split_rows[split], key=lambda record: record["record_id"])
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in rows:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        entries.append(
            {
                "name": split,
                "path": path.name,
                "role": "tool_sft",
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "record_count": len(rows),
                "schema_version": BANKING_TOOL_SFT_CONTRACT,
                "licenses": ["MIT"],
            }
        )
    return entries


def _render_data_card(report: dict[str, Any]) -> str:
    summary = report["summary"]
    splits = report["splits"]
    checks = report["checks"]
    path_lines = "\n".join(
        f"- `{name}`: {count}" for name, count in sorted(checks["paths"].items())
    )
    return f"""---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- banking
- tool-calling
- synthetic
- conversational
pretty_name: Retail Bank Agent Tool-Use SFT
---

# Retail Bank Agent Tool-Use SFT

This dataset contains {summary["total_records"]:,} deterministic, fictional
retail-banking conversations for supervised fine-tuning of a conversational
tool-using model.

## Splits

- Train: {splits["train"]["records"]:,}
- Validation: {splits["validation"]["records"]:,}
- Test: {splits["test"]["records"]:,}
- Corpus fingerprint: `{summary["corpus_fingerprint"]}`
- Split seed: `{summary["split_seed"]}`

## Coverage

The corpus covers all nine public synthetic-bank tools, successful and failed
tool results, clarification, general banking FAQ, hard-negative private-field
requests, out-of-domain refusal, multi-turn context, and ordered multi-tool
calls.

{path_lines}

Every tool-bearing record was replayed against an isolated synthetic bank state
before inclusion. Assistant tool-call and final-response tokens are trainable;
system, user, and tool-result tokens are context only.

## Source policy

All included rows are self-authored synthetic data under MIT. Banking77 and
CLINC are classifier/evaluation-only and contribute no generative SFT rows.
Bitext data remains quarantined and contributes no rows.

This dataset contains no real customers, credentials, accounts, or financial
events. It is for a research demonstration, not production banking.
"""


def _message(role: str, content: str, *, loss: bool) -> dict[str, Any]:
    return {"role": role, "content": content, "loss": loss}


def _format_usd(cents: Any) -> str:
    return f"{int(cents) / 100:,.2f}"


def _realize_user(template: Scenario, occurrence: int) -> str:
    stems = _user_stems(template)
    stem = _pick(stems, occurrence)
    if template.scenario_family in {
        "action_summary_followup",
        "no_action_followup",
        "small_talk_greeting",
        "small_talk_checkin",
        "conversational_thanks",
    }:
        qualifier = _pick(
            (
                "",
                "I want to be sure.",
                "Just so I understand.",
                "Please be clear.",
                "I’m checking the conversation.",
                "Before we continue.",
                "In this chat.",
                "For my records.",
                "One quick question.",
                "I want to confirm.",
            ),
            occurrence // len(stems),
        )
        closer = _pick(
            (
                "",
                "Thanks.",
                "Keep it concise.",
                "That is all I need.",
                "I am following along.",
                "I just want to confirm.",
                "Please answer naturally.",
                "Then we can continue.",
            ),
            occurrence // (len(stems) * 10),
        )
        return " ".join(part for part in (stem, qualifier, closer) if part).strip()
    opener = _pick(REALIZER_OPENERS, occurrence // len(stems))
    closer = _pick(REALIZER_CLOSERS, occurrence // (len(stems) * len(REALIZER_OPENERS)))
    context = _natural_context(template, occurrence)
    pieces = [opener, stem]
    if closer:
        pieces.append(closer)
    pieces.append(context)
    return " ".join(piece.strip() for piece in pieces if piece.strip())


def _realize_final(template: Scenario, occurrence: int) -> str:
    if not template.tool_plan:
        return template.final_response.strip()
    prefix = _pick(REALIZER_FINAL_PREFIXES, occurrence)
    return f"{prefix} {template.final_response}".strip()


def _user_stems(template: Scenario) -> tuple[str, ...]:
    family = template.scenario_family
    card_last4 = _tool_arg(template, "last4", "the card")
    merchant = _tool_arg(template, "description", "that merchant")
    recipient = _tool_arg(template, "recipient", "that recipient")
    if family == "read_accounts":
        return (
            "show the accounts available to me and their balances",
            "pull up my account list with balances",
            "tell me which deposit accounts are on this profile",
            "check my checking and savings account summary",
            "look up my signed-in account balances",
            "review the accounts connected to this login",
            "summarize my open banking accounts",
            "list my accounts with their ending digits",
        )
    if family == "read_cards":
        return (
            "check whether my debit card is active",
            "show the status of my card",
            "look up my card status for this profile",
            "tell me if the card on file can be used",
            "review my debit card and wallet status",
            "confirm the current card state",
            "pull my card details from the banking tools",
            "check the card ending digits on this login",
        )
    if family == "read_service_cases":
        return (
            "show any service cases on my profile",
            "check whether I have support cases",
            "look up my recent case history",
            "review open or closed service requests",
            "tell me what support cases are recorded",
            "pull the latest service case list",
            "check my banking support case status",
            "summarize my recent service cases",
        )
    if family == "read_transactions":
        return (
            "show my three most recent transactions",
            "pull the latest three account transactions",
            "review my recent transaction history",
            "list the newest three transactions on my checking account",
            "check the latest activity on my account",
            "show recent debits and credits from my profile",
            "look up my latest posted transactions",
            "summarize the newest three transaction entries",
        )
    if family == "read_transfers":
        return (
            "show the transfers on my account",
            "check my transfer history and status",
            "list recent transfers for this profile",
            "pull my pending and completed transfers",
            "review transfers connected to my checking account",
            "tell me which transfers are recorded",
            "look up the transfer list",
            "summarize my transfer activity",
        )
    if family == "card_freeze":
        return (
            f"freeze my debit card ending in {card_last4}",
            f"put a freeze on the card ending {card_last4}",
            f"lock the debit card with last four {card_last4}",
            f"turn off card {card_last4} for purchases",
            f"freeze the active card ending {card_last4}",
            f"stop use of my card ending in {card_last4}",
            f"secure the debit card with digits {card_last4}",
            f"place a freeze on card {card_last4}",
        )
    if family == "card_replace":
        return (
            f"replace my card ending in {card_last4}",
            f"request a replacement for card {card_last4}",
            f"start a replacement card order for {card_last4}",
            f"mark the card ending {card_last4} for replacement",
            f"get a new debit card for last four {card_last4}",
            f"replace the debit card ending {card_last4}",
            f"set up replacement service for card {card_last4}",
            f"request a new card for digits {card_last4}",
        )
    if family == "transaction_dispute":
        return (
            f"dispute the {merchant} debit",
            f"open a dispute for {merchant}",
            f"challenge the charge from {merchant}",
            f"report the {merchant} transaction as disputed",
            f"file a dispute on that {merchant} purchase",
            f"start a transaction dispute for {merchant}",
            f"mark the {merchant} debit for review",
            f"submit a dispute for the {merchant} charge",
        )
    if family == "transfer_cancel":
        return (
            f"cancel my pending transfer to {recipient}",
            f"stop the {recipient} transfer",
            f"cancel the transfer headed to {recipient}",
            f"void my pending {recipient} payment",
            f"remove the scheduled transfer to {recipient}",
            f"call off the {recipient} transfer",
            f"cancel the pending payment for {recipient}",
            f"stop that pending transfer for {recipient}",
        )
    if family == "multi_tool_card_action":
        return (
            "check my card status and then freeze the active debit card",
            "look at my card list before freezing the active card",
            "find the active card and place a freeze on it",
            "review my cards, then lock the active debit card",
            "pull my card details and freeze the active one",
            "check which card is active and secure it",
            "list my cards first, then freeze the debit card",
            f"verify the card ending digits before freezing card {card_last4}",
        )
    if family == "emergency_card_freeze":
        return (
            "my card was stolen freeze it",
            "I lost my debit card lock it now",
            "someone took my card freeze the active one",
            "my wallet is gone secure my debit card",
            "I cannot find my card please freeze it",
            "my debit card is missing lock it",
            "the card was stolen please secure it",
            "freeze whichever debit card is active because I lost mine",
            "my wallet was stolen find the active card and freeze it",
            "I left my card somewhere lock it before it is used",
            "secure my active debit card I think it was taken",
            "my card is gone please find it and freeze it",
        )
    if family == "action_summary_followup":
        return (
            "what did you just do",
            "what action did you take",
            "can you summarize what you changed",
            "did you actually freeze the card",
            "which card did you freeze",
            "remind me what you completed",
            "what happened in the previous step",
            "tell me exactly what was done",
        )
    if family == "no_action_followup":
        return (
            "did you replace it already",
            "was any card changed",
            "what did you do so far",
            "did that request complete",
            "have you taken any action yet",
            "did you order the replacement",
            "was the card status changed",
            "summarize what happened",
        )
    if family == "clarification_card":
        return (
            "replace my card",
            "send me a new card",
            "start a card replacement",
            "get my debit card replaced",
            "replace the card on my profile",
            "order a replacement card",
            "help with a card replacement",
            "set up a new card for me",
        )
    if family == "no_tool_banking_faq":
        return (
            "explain how overdraft fees usually work",
            "tell me what overdraft fees mean",
            "describe overdraft charges in general",
            "explain when overdraft fees can happen",
            "give a general overview of overdrafts",
            "help me understand overdraft fee basics",
            "answer a general overdraft policy question",
            "summarize how overdraft charges are usually handled",
        )
    if family == "faq_mortgage":
        return (
            "can you help me open a mortgage account",
            "how do I apply for a mortgage",
            "explain the usual mortgage application steps",
            "can this demo approve a home loan",
            "what is involved in getting a mortgage",
            "tell me how mortgage applications generally work",
            "can you start a mortgage for me",
            "what should I expect when applying for a home loan",
        )
    if family == "faq_mortgage_age":
        return (
            "how old do I need to be to apply for a mortgage",
            "what is the minimum age for a home loan",
            "do mortgage applicants need to be at least 18",
            "can someone under 18 apply for a mortgage",
            "explain the usual age requirement for a mortgage",
            "what age must a borrower be for a home loan",
            "is there a legal adult requirement for mortgages",
            "tell me the typical minimum mortgage application age",
        )
    if family == "faq_deposit_opening":
        return (
            "how would I open a new savings account",
            "can you open another checking account for me",
            "what are the usual steps to open a bank account",
            "can this demo create a new deposit account",
            "explain how account opening generally works",
            "what do banks usually require for a new account",
            "help me understand opening a savings account",
            "can I add a new checking account in this chat",
        )
    if family == "faq_savings_interest":
        return (
            "how does interest on a savings account work",
            "explain savings interest in general",
            "what does annual percentage yield mean for savings",
            "how do banks calculate savings interest",
            "when is savings interest usually credited",
            "help me understand interest and compounding",
            "what affects the interest earned on savings",
            "give me a general explanation of savings rates",
        )
    if family == "small_talk_greeting":
        return (
            "yo sup",
            "hello there",
            "hey",
            "good morning",
            "hi how is it going",
            "hey bank bot",
            "hello can you help",
            "hi I just logged in",
        )
    if family == "small_talk_checkin":
        return (
            "how are you doing",
            "are you ready to help",
            "what can you help me with",
            "can we talk about my banking",
            "are you the banking assistant",
            "what do you do here",
            "is anyone there",
            "can you assist me today",
        )
    if family == "conversational_thanks":
        return (
            "thanks for the help",
            "thank you",
            "that was helpful",
            "great thanks",
            "I appreciate it",
            "thanks that answers my question",
            "perfect thank you",
            "okay thanks",
        )
    if family == "ood":
        return (
            "tell me whether it will rain in Seattle tomorrow",
            "give me tomorrow's Seattle weather forecast",
            "check if Seattle has rain coming tomorrow",
            "answer a weather question about Seattle",
            "forecast tomorrow's rain in Seattle",
            "look up the Seattle weather for tomorrow",
            "help with tomorrow's Seattle rain chances",
            "tell me the weather outside banking",
        )
    if family == "hard_negative_private_id":
        return (
            "give me my full account number and customer ID",
            "show the private account number for this login",
            "tell me the customer identifier behind my profile",
            "reveal my complete account number",
            "send my full account and customer IDs in chat",
            "display the private banking identifiers",
            "provide all backend IDs for my account",
            "share my complete account number",
        )
    if family == "backend_error" and template.template_id == "cancel-completed-v1":
        return (
            f"cancel my transfer to {recipient}",
            f"stop the {recipient} transfer",
            f"try to cancel the payment to {recipient}",
            f"remove the {recipient} transfer from my account",
            f"cancel that completed {recipient} transfer",
            f"void the transfer for {recipient}",
            f"reverse my {recipient} transfer if possible",
            f"call off the {recipient} payment",
        )
    if family == "backend_error":
        return (
            f"dispute the {merchant} charge",
            f"open a dispute for {merchant}",
            f"challenge a {merchant} debit",
            f"file a dispute on {merchant}",
            f"report {merchant} as an unauthorized charge",
            f"start a charge review for {merchant}",
            f"mark {merchant} for dispute",
            f"submit a transaction dispute for {merchant}",
        )
    if family == "multi_turn_dispute":
        return (
            f"it was {merchant}",
            f"the merchant was {merchant}",
            f"that charge came from {merchant}",
            f"use {merchant} for the dispute",
            f"the transaction name is {merchant}",
            f"{merchant} is the one",
            f"please dispute {merchant}",
            f"the debit I meant was {merchant}",
        )
    return (template.user,)


def _natural_context(template: Scenario, seed: int) -> str:
    context = _pick(REALIZER_CONTEXTS, seed)
    if not context:
        return ""
    if template.scenario_family in {"transfer_cancel", "read_transfers"}:
        return f"{context} for an upcoming bill"
    if template.scenario_family in {"transaction_dispute", "multi_turn_dispute"}:
        return f"{context} after reviewing recent purchases"
    if template.scenario_family in {
        "card_freeze",
        "card_replace",
        "multi_tool_card_action",
        "emergency_card_freeze",
    }:
        return f"{context} after checking my wallet"
    return context


def _tool_arg(template: Scenario, key: str, default: str) -> str:
    for call in template.tool_plan:
        if key in call.arguments:
            return str(call.arguments[key])
    return default


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def _last4(seed: int, salt: int) -> str:
    return f"{1000 + ((seed + salt * 997) % 9000):04d}"


def _pick(values: tuple[str, ...], seed: int) -> str:
    return values[seed % len(values)]


def _last_user_message(record: dict[str, Any]) -> dict[str, Any]:
    for message in reversed(record.get("messages", [])):
        if message.get("role") == "user":
            return message
    raise BankingToolSftDataError(f"{record.get('record_id')} has no user message")


def _final_assistant_message(record: dict[str, Any]) -> dict[str, Any]:
    for message in reversed(record.get("messages", [])):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            return message
    raise BankingToolSftDataError(f"{record.get('record_id')} has no final assistant message")


def _immutable_record_hash(record: dict[str, Any]) -> str:
    immutable_messages = []
    messages = list(record.get("messages", []))
    last_user_index = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        default=-1,
    )
    final_assistant_index = max(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant" and not message.get("tool_calls")
        ),
        default=-1,
    )
    for index, message in enumerate(messages):
        if index in {last_user_index, final_assistant_index}:
            continue
        immutable_messages.append(message)
    payload = {
        "schema_version": record.get("schema_version"),
        "record_id": record.get("record_id"),
        "messages_except_final_assistant": immutable_messages,
        "expected": record.get("expected"),
        "split_keys": record.get("split_keys"),
        "validation": {
            "tool_manifest_hash": record.get("validation", {}).get("tool_manifest_hash"),
            "replay_hash": record.get("validation", {}).get("replay_hash"),
        },
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def _split_group(scenario: Scenario) -> str:
    values = (
        scenario.scenario_family,
        scenario.state_seed,
        scenario.customer_id,
        scenario.template_id,
    )
    return "|".join(values)


def _tool_manifest_hash() -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(public_tool_manifest())).hexdigest()}"


def _safe_error_message(message: str) -> str:
    lowered = message.lower()
    if "matching synthetic" in lowered or "no matching" in lowered:
        return "No matching eligible synthetic banking record was found."
    if "unsupported" in lowered:
        return "The requested tool arguments are not supported."
    return "The synthetic banking action could not be completed."


def _stable_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    stable = json.loads(json.dumps(snapshot, sort_keys=True))
    normalized_cases = []
    generated_index = 0
    for case in stable.get("service_cases", []):
        case_id = str(case.get("case_id", ""))
        if case_id.startswith("case_") and not case_id.startswith(("case_alex_", "case_maya_")):
            generated_index += 1
            case["case_id"] = f"generated_case_{generated_index:03d}"
            case["created_at"] = "generated-timestamp"
        normalized_cases.append(case)
    stable["service_cases"] = normalized_cases
    return stable


def _resolve_bank_path(path: Path) -> Path:
    if path.exists():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / path
    if candidate.exists():
        return candidate
    raise BankingToolSftDataError(f"synthetic bank file not found: {path}")


def _load_bank_registry(bank_path: Path, *, payload: dict[str, Any] | None = None) -> Any:
    module_path = bank_path.parent / "mock_bank.py"
    module_name = "banking_tool_sft_mock_bank"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise BankingToolSftDataError(f"cannot import mock bank from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if payload is not None:
        return module.SessionBankRegistry(payload, max_sessions=4)
    return module.SessionBankRegistry.from_json(bank_path, max_sessions=4)


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise BankingToolSftDataError(f"missing required string field {key}")
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    raise SystemExit(main())
