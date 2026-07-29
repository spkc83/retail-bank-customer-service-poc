from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from hello_slm.banking_tool_eval import (
    StaticPredictionModel,
    TaggedJsonToolAdapter,
    evaluate_records,
    fingerprint_records,
    state_hash,
)
from hello_slm.banking_tool_sft_data import generate_records


def _record(
    record_id: str,
    *,
    expected: dict,
    messages: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": "banking-tool-eval/v1",
        "record_id": record_id,
        "messages": messages or [{"role": "user", "content": "test"}],
        "initial_state": {
            "cards": [{"last4": "4821", "status": "active"}],
            "transfers": [{"recipient": "River Consulting", "status": "pending"}],
        },
        "expected": expected,
    }


def test_tool_name_and_argument_metrics_use_exact_ordered_denominators() -> None:
    records = [
        _record(
            "match",
            expected={
                "requires_tool": True,
                "tool_calls": [
                    {"name": "list_cards", "arguments": {}},
                    {"name": "freeze_card", "arguments": {"last4": "4821"}},
                ],
                "multi_tool": True,
            },
        ),
        _record(
            "wrong_order",
            expected={
                "requires_tool": True,
                "tool_calls": [
                    {"name": "list_cards", "arguments": {}},
                    {"name": "freeze_card", "arguments": {"last4": "4821"}},
                ],
                "multi_tool": True,
            },
        ),
    ]
    model = StaticPredictionModel(
        {
            "match": (
                '<tool_call>{"name":"list_cards","arguments":{}}</tool_call>'
                '<tool_call>{"name":"freeze_card","arguments":{"last4":"4821"}}</tool_call>'
            ),
            "wrong_order": (
                '<tool_call>{"name":"freeze_card","arguments":{"last4":"4821"}}</tool_call>'
                '<tool_call>{"name":"list_cards","arguments":{}}</tool_call>'
            ),
        }
    )

    report = evaluate_records(records, model=model, adapter=TaggedJsonToolAdapter())

    assert report["metrics"]["tool_name_accuracy"]["numerator"] == 2
    assert report["metrics"]["tool_name_accuracy"]["denominator"] == 4
    assert report["metrics"]["tool_argument_accuracy"]["numerator"] == 2
    assert report["metrics"]["tool_argument_accuracy"]["denominator"] == 4
    assert report["metrics"]["multi_tool_exact_sequence"]["numerator"] == 1
    assert report["metrics"]["multi_tool_exact_sequence"]["denominator"] == 2


def test_parse_failures_and_private_arguments_are_reported_with_call_denominators() -> None:
    records = [
        _record(
            "malformed",
            expected={
                "requires_tool": True,
                "tool_calls": [{"name": "list_accounts", "arguments": {}}],
            },
        ),
        _record(
            "private_arg",
            expected={
                "requires_tool": True,
                "tool_calls": [
                    {
                        "name": "cancel_transfer",
                        "arguments": {"recipient": "River Consulting"},
                    }
                ],
            },
        ),
    ]
    model = StaticPredictionModel(
        {
            "malformed": "<tool_call>not-json</tool_call>",
            "private_arg": (
                '<tool_call>{"name":"cancel_transfer",'
                '"arguments":{"transfer_id":"trf_alex_100"}}</tool_call>'
            ),
        }
    )

    report = evaluate_records(records, model=model, adapter=TaggedJsonToolAdapter())

    assert report["parse_failures"] == 1
    assert report["metrics"]["malformed_tool_call_rate"]["numerator"] == 1
    assert report["metrics"]["malformed_tool_call_rate"]["denominator"] == 2
    assert report["metrics"]["unsupported_private_arguments"]["numerator"] == 1
    assert report["metrics"]["unsupported_private_arguments"]["denominator"] == 1
    assert report["records"]["private_arg"]["manifest_failures"] == [
        "cancel_transfer unsupported/private args: ['transfer_id']"
    ]


def test_replay_final_state_grounding_faq_credentials_and_ood_metrics() -> None:
    final_hash = state_hash(
        {
            "cards": [{"last4": "4821", "status": "frozen"}],
            "transfers": [{"recipient": "River Consulting", "status": "pending"}],
        }
    )
    records = [
        _record(
            "freeze",
            expected={
                "requires_tool": True,
                "tool_calls": [{"name": "freeze_card", "arguments": {"last4": "4821"}}],
                "executable": True,
                "final_state_hash": final_hash,
                "grounding_facts": ["4821", "frozen"],
                "forbidden_facts": ["replacement pending"],
            },
        ),
        _record(
            "faq",
            expected={
                "requires_tool": False,
                "response_path": "faq",
                "faq_facts": ["mobile app", "card services"],
                "forbidden_facts": ["account number"],
            },
        ),
        _record(
            "clarify",
            expected={
                "requires_tool": False,
                "response_path": "clarification",
                "clarification_missing_field": "last four digits",
            },
        ),
        _record(
            "ood",
            expected={
                "requires_tool": False,
                "response_path": "ood",
                "path_markers": ["retail banking"],
            },
        ),
    ]
    model = StaticPredictionModel(
        {
            "freeze": (
                '<tool_call>{"name":"freeze_card","arguments":{"last4":"4821"}}</tool_call>\n'
                "Your card ending in 4821 is frozen."
            ),
            "faq": "Use the mobile app and open card services.",
            "clarify": "Which card should I freeze? Please provide the last four digits.",
            "ood": "I can only help with retail banking questions.",
        }
    )

    report = evaluate_records(records, model=model, adapter=TaggedJsonToolAdapter())

    assert report["metrics"]["executable_tool_success"]["numerator"] == 1
    assert report["metrics"]["executable_tool_success"]["denominator"] == 1
    assert report["metrics"]["grounded_final_factuality"]["numerator"] == 1
    assert report["metrics"]["grounded_final_factuality"]["denominator"] == 1
    assert report["metrics"]["no_tool_faq_quality"]["numerator"] == 1
    assert report["metrics"]["no_tool_faq_quality"]["denominator"] == 1
    assert report["metrics"]["clarification_appropriateness"]["numerator"] == 1
    assert report["metrics"]["clarification_appropriateness"]["denominator"] == 1
    assert report["metrics"]["ood_small_talk_response_path"]["numerator"] == 1
    assert report["metrics"]["ood_small_talk_response_path"]["denominator"] == 1
    assert report["metrics"]["credential_request_rate"]["numerator"] == 0
    assert report["metrics"]["credential_request_rate"]["denominator"] == 4


def test_report_includes_fingerprints_and_record_parse_failure_details() -> None:
    records = [
        _record(
            "missing",
            expected={
                "requires_tool": True,
                "tool_calls": [{"name": "list_accounts", "arguments": {}}],
            },
        )
    ]
    report = evaluate_records(
        records,
        model=StaticPredictionModel({"missing": "I need your account number first."}),
        adapter=TaggedJsonToolAdapter(template_hash="sha256:test-template"),
        checkpoint_revision="local-test",
    )

    assert report["dataset_fingerprint"] == fingerprint_records(records)
    assert report["adapter_template_hash"] == "sha256:test-template"
    assert report["checkpoint_revision"] == "local-test"
    assert report["records"]["missing"]["tool_name_accuracy"] is False
    assert report["records"]["missing"]["credential_request"] is True


def test_cli_dry_run_writes_json_report(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/banking_v2/evaluate_tool_model.py",
            "--dry-run",
            "--output",
            str(output_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "dataset_fingerprint" in completed.stdout
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["metrics"]["tool_name_accuracy"]["denominator"] >= 1
    assert report["parse_failures"] == 0


def test_generated_sft_records_have_evaluable_expected_tool_calls() -> None:
    records = generate_records(pilot_count=18)
    tool_records = [
        record for record in records if record["expected"]["requires_tool"]
    ]
    outputs = {}
    for record in records:
        calls = record["expected"]["tool_calls"]
        outputs[record["record_id"]] = "".join(
            "<tool_call>"
            + json.dumps(
                {"name": call["name"], "arguments": call["arguments"]},
                separators=(",", ":"),
            )
            + "</tool_call>"
            for call in calls
        ) or str(record["messages"][-1]["content"])

    report = evaluate_records(
        records,
        model=StaticPredictionModel(outputs),
        adapter=TaggedJsonToolAdapter(),
    )

    expected_denominator = sum(
        len(record["expected"]["tool_calls"]) for record in tool_records
    )
    assert expected_denominator > 0
    assert report["metrics"]["tool_name_accuracy"]["denominator"] == expected_denominator
    assert report["metrics"]["tool_argument_accuracy"]["denominator"] == expected_denominator
