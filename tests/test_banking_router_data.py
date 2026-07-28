from __future__ import annotations

from collections import Counter

from hello_slm.banking_router_data import (
    CLINC_SUPPORTED_BANKING_LABELS,
    build_router_splits,
    normalize_router_text,
    render_router_input,
)


def banking_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for intent in ("card_arrival", "pending_transfer"):
        for index in range(12):
            rows.append(
                {
                    "text": f"{intent} training example {index}",
                    "label": intent,
                    "split": "train",
                    "source_row_id": len(rows),
                }
            )
        for index in range(3):
            rows.append(
                {
                    "text": f"{intent} untouched test example {index}",
                    "label": intent,
                    "split": "test",
                    "source_row_id": len(rows),
                }
            )
    return rows


def clinc_payload() -> dict[str, list[list[str]]]:
    return {
        "train": [
            ["my card was declined", "card_declined"],
            ["tell me a joke", "tell_joke"],
        ],
        "val": [
            ["my account is frozen", "freeze_account"],
            ["what is the weather", "weather"],
        ],
        "test": [
            ["where is my new card", "new_card"],
            ["play some music", "play_music"],
        ],
        "oos_train": [["explain photosynthesis", "oos"]],
        "oos_val": [["who painted this", "oos"]],
        "oos_test": [["how tall are penguins", "oos"]],
    }


def test_router_input_places_current_turn_before_previous_context() -> None:
    rendered = render_router_input(
        "What happened now?",
        previous_user="My card transfer failed.",
    )

    assert rendered == (
        "[CURRENT]\nWhat happened now?\n"
        "[PREVIOUS_USER]\nMy card transfer failed."
    )


def test_build_router_splits_preserves_test_and_labels_clinc_boundaries() -> None:
    splits, report = build_router_splits(
        banking_rows(),
        clinc_payload(),
        validation_fraction=0.25,
        seed=7101,
    )

    assert set(splits) == {"train", "validation", "test"}
    assert {
        row["current_text"]
        for row in splits["test"]
        if row["example_kind"] == "banking77_single"
    } == {
        "card_arrival untouched test example 0",
        "card_arrival untouched test example 1",
        "card_arrival untouched test example 2",
        "pending_transfer untouched test example 0",
        "pending_transfer untouched test example 1",
        "pending_transfer untouched test example 2",
    }

    clinc_rows = [
        row
        for split_rows in splits.values()
        for row in split_rows
        if str(row["example_kind"]).startswith("clinc_")
    ]
    by_label = {str(row["source_label"]): row for row in clinc_rows}
    assert "card_declined" in CLINC_SUPPORTED_BANKING_LABELS
    assert by_label["card_declined"]["domain_label"] == 1
    assert by_label["card_declined"]["intent_label"] == -100
    assert by_label["tell_joke"]["domain_label"] == 0
    assert by_label["oos"]["domain_label"] == 0

    assert report["banking77_test_rows"] == 6
    assert report["intent_count"] == 2
    assert report["pii_matches"] == 0


def test_transition_examples_teach_current_turn_priority_without_split_leakage() -> None:
    splits, report = build_router_splits(
        banking_rows(),
        clinc_payload(),
        validation_fraction=0.25,
        seed=7101,
    )

    transitions = [
        row
        for split_rows in splits.values()
        for row in split_rows
        if row["example_kind"] == "banking_to_ood_transition"
    ]
    assert transitions
    assert all(row["domain_label"] == 0 for row in transitions)
    assert all("[PREVIOUS_USER]" in str(row["text"]) for row in transitions)

    normalized_by_split = {
        split: {normalize_router_text(str(row["text"])) for row in rows}
        for split, rows in splits.items()
    }
    assert normalized_by_split["train"].isdisjoint(normalized_by_split["validation"])
    assert normalized_by_split["train"].isdisjoint(normalized_by_split["test"])
    assert normalized_by_split["validation"].isdisjoint(normalized_by_split["test"])
    assert report["cross_split_duplicates_removed"] >= 0

    kinds = Counter(row["example_kind"] for row in splits["train"])
    assert kinds["same_intent_followup"] > 0
    assert kinds["banking_to_ood_transition"] > 0
