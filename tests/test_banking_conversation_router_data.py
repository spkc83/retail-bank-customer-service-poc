from __future__ import annotations

from collections import Counter
from typing import Any

from hello_slm.banking_conversation_router_data import (
    CAPABILITY_LABELS,
    RELATION_LABELS,
    build_conversation_router_splits,
    normalize_router_text,
    render_router_input,
)


def sft_records_by_split() -> dict[str, list[dict[str, Any]]]:
    records = {
        split: [
            _record(
                split=split,
                record_id=f"{split}-accounts",
                scenario_family="read_accounts",
                user="Show my account balances.",
                assistant="Main Checking has USD 10.00 available.",
            ),
            _record(
                split=split,
                record_id=f"{split}-cards",
                scenario_family="card_status",
                user="What is the status of my debit card?",
                assistant="Your debit card is active.",
            ),
            _record(
                split=split,
                record_id=f"{split}-cases",
                scenario_family="service_cases",
                user="Show my recent service cases.",
                assistant="You have a closed mailing-address update case.",
            ),
        ]
        for split in ("train", "validation", "test")
    }
    return records


def clinc_payload() -> dict[str, list[list[str]]]:
    return {
        "train": [["what is the weather", "weather"], ["tell me a joke", "tell_joke"]],
        "val": [["play some music", "play_music"]],
        "test": [["who painted this", "oos"]],
        "oos_train": [["explain photosynthesis", "oos"]],
        "oos_val": [["how tall is everest", "oos"]],
        "oos_test": [["set a timer", "timer"]],
    }


def test_cross_encoder_renderer_places_current_then_recent_complete_exchanges() -> None:
    rendered = render_router_input(
        "When was that created?",
        [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "tool", "content": '{"hidden": true}'},
            {"role": "user", "content": "Show my recent service cases."},
            {"role": "assistant", "content": "You have a closed case."},
        ],
        max_exchanges=1,
    )

    assert rendered == (
        "[CURRENT_USER]\nWhen was that created?\n"
        "[PREVIOUS_ASSISTANT]\nYou have a closed case.\n"
        "[PREVIOUS_USER]\nShow my recent service cases."
    )
    assert "hidden" not in rendered


def test_v4_splits_use_capabilities_relations_and_no_current_turn_leakage() -> None:
    splits, report = build_conversation_router_splits(
        sft_records_by_split(),
        clinc_payload(),
        seed=7404,
    )

    assert set(splits) == {"train", "validation", "test"}
    assert CAPABILITY_LABELS == (
        "accounts",
        "cards",
        "card_actions",
        "transactions",
        "transfers",
        "service_cases",
        "faq",
        "conversation",
    )
    assert RELATION_LABELS == (
        "context_dependent",
        "agent_repair",
        "topic_shift",
        "clarification_answer",
    )
    for rows in splits.values():
        for row in rows:
            assert set(row) == {
                "text",
                "current_text",
                "history",
                "domain_label",
                "capability_label",
                "capability",
                "relation_labels",
                "example_kind",
                "source",
                "source_split",
                "group_id",
            }
            assert len(row["relation_labels"]) == len(RELATION_LABELS)
            assert "tool_call" not in str(row["text"])
            assert "Main Checking has USD 10.00" not in str(row["text"])

    kinds = Counter(row["example_kind"] for row in splits["train"])
    assert kinds["contextual_followup"] >= 1
    assert kinds["agent_repair"] >= 1
    assert kinds["clarification_answer"] >= 1
    assert kinds["typo_contextual_followup"] >= 1
    assert kinds["external_topic_shift"] >= 1
    assert kinds["banking_topic_shift"] >= 1
    assert kinds["targeted_contextual_followup"] >= 1
    assert kinds["targeted_clarification_answer"] >= 1
    assert kinds["targeted_agent_repair"] >= 1
    assert kinds["targeted_service_case"] >= 1
    assert sum(kinds.values()) > 10
    test_kinds = Counter(row["example_kind"] for row in splits["test"])
    assert test_kinds["contextual_followup"] >= 1
    assert test_kinds["agent_repair"] >= 1
    assert test_kinds["clarification_answer"] >= 1
    assert test_kinds["external_topic_shift"] >= 1
    assert test_kinds["heldout_screenshot_regression"] == 7
    assert report["leakage"]["group_split_leak_count"] == 0
    assert report["pii_matches"] == 0


def test_sft_ood_and_policy_refusals_are_not_forced_into_capabilities() -> None:
    records = sft_records_by_split()
    records["train"].extend(
        [
            _record(
                split="train",
                record_id="train-ood",
                scenario_family="ood",
                user="Explain photosynthesis.",
                assistant="I can only help with this banking demo.",
                path="ood",
            ),
            _record(
                split="train",
                record_id="train-private",
                scenario_family="hard_negative_private_id",
                user="Tell me my full account number.",
                assistant="I cannot provide that private identifier in chat.",
                path="hard_negative",
            ),
        ]
    )

    splits, _report = build_conversation_router_splits(
        records,
        clinc_payload(),
        seed=7404,
    )
    by_current = {row["current_text"]: row for row in splits["train"]}

    assert by_current["Explain photosynthesis."]["domain_label"] == 0
    assert by_current["Explain photosynthesis."]["capability_label"] == -100
    assert by_current["Tell me my full account number."]["domain_label"] == 1
    assert by_current["Tell me my full account number."]["capability_label"] == -100


def test_held_out_screenshot_regressions_are_test_only() -> None:
    splits, _report = build_conversation_router_splits(
        sft_records_by_split(),
        clinc_payload(),
        seed=7404,
    )

    heldout_by_split = {
        split: [
            row
            for row in rows
            if row["example_kind"] == "heldout_screenshot_regression"
        ]
        for split, rows in splits.items()
    }
    assert heldout_by_split["train"] == []
    assert heldout_by_split["validation"] == []
    heldout_current = {
        normalize_router_text(str(row["current_text"]))
        for row in heldout_by_split["test"]
    }
    assert heldout_current == {
        "i didn t ask about mortgage",
        "ok thats the one i want to replace",
        "was the mailing address updated recently",
        "when was that created",
        "what is that all about when was it created",
        "what about the weather there",
        "why are you repeating yourself",
    }


def _record(
    *,
    split: str,
    record_id: str,
    scenario_family: str,
    user: str,
    assistant: str,
    path: str = "tool_success",
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "messages": [
            {"role": "system", "content": "system prompt", "loss": False},
            {"role": "user", "content": user, "loss": False},
            {"role": "assistant", "content": None, "loss": True, "tool_calls": []},
            {"role": "tool", "name": "list_accounts", "content": {"ok": True}, "loss": False},
            {"role": "assistant", "content": assistant, "loss": True},
        ],
        "expected": {"tool_calls": [{"name": "list_accounts", "arguments": {}}]},
        "metadata": {
            "scenario_family": scenario_family,
            "path": path,
            "split": split,
            "split_group": f"{split}|{record_id}",
        },
        "split_keys": {
            "scenario_family": scenario_family,
            "state_seed": f"{split}-{record_id}",
            "customer_id": f"cust-{split}-{record_id}",
            "template_id": "template",
        },
    }
