from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hello_slm.banking_data import (  # noqa: E402
    BANKING_V2_CANNED_OOD_RESPONSE,
    BankingDataError,
    SourceSnapshot,
    normalize_bitext_placeholders,
    prepare,
    scrub_pii_like,
    validate_banking_v2_manifest,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_snapshot(path: Path) -> None:
    rows = [
        {
            "source_row_id": 0,
            "tags": "B",
            "instruction": "I need help activating my {{Credit Card}}.",
            "category": "CARD",
            "intent": "activate_card",
            "response": "Open {{Banking App}} and choose {{Card Services}}.",
        },
        {
            "source_row_id": 1,
            "tags": "B",
            "instruction": "Please help me activate the credit card.",
            "category": "CARD",
            "intent": "activate_card",
            "response": "Use the banking app card services menu.",
        },
        {
            "source_row_id": 2,
            "tags": "B",
            "instruction": "How do I make a bank transfer?",
            "category": "TRANSFER",
            "intent": "make_transfer",
            "response": (
                "Use transfers, add a recipient, review the details, and confirm. "
                "Do not enter 4111 1111 1111 1111 in chat."
            ),
        },
        {
            "source_row_id": 3,
            "tags": "B",
            "instruction": "Where can I find an ATM?",
            "category": "ATM",
            "intent": "find_ATM",
            "response": "Use the ATM locator in online or mobile banking.",
        },
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def test_placeholder_normalization_removes_unresolved_tokens() -> None:
    text = "Call {{Customer Support Phone Number}} during {{Customer Support Working Hours}}."

    normalized, count = normalize_bitext_placeholders(text)

    assert count == 2
    assert "{{" not in normalized
    assert "customer support phone number" not in normalized.lower()
    assert "customer support" in normalized.lower()


def test_pii_like_scrubber_removes_common_patterns() -> None:
    scrubbed, replacements = scrub_pii_like(
        "Email me at user@example.com or call 212-555-0199 about 4111 1111 1111 1111."
    )

    assert replacements == 3
    assert "user@example.com" not in scrubbed
    assert "212-555-0199" not in scrubbed
    assert "4111 1111 1111 1111" not in scrubbed


def test_prepare_generates_banking_v2_dataset_and_report(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "bitext-source.jsonl"
    _write_snapshot(snapshot_path)
    polyai_snapshot = tmp_path / "polyai-source.jsonl"
    polyai_snapshot.write_text(
        json.dumps(
            {
                "source_row_id": 0,
                "split": "test",
                "text": "How do I activate my card?",
                "label": "activate_my_card",
                "source_dataset": "PolyAI/banking77",
                "source_revision": "rev-polyai",
                "license": "CC-BY-4.0",
                "trainable": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "banking-v2"
    lock_path = tmp_path / "banking-v2.lock.json"

    report = prepare(
        snapshot=SourceSnapshot(
            path=snapshot_path,
            dataset_id="bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
            revision="rev-bitext",
            license="CDLA-Sharing-1.0",
            sha256="0" * 64,
        ),
        output_dir=output_dir,
        lock_path=lock_path,
        split_seed=7101,
        polyai_snapshot=polyai_snapshot,
    )
    second_report = prepare(
        snapshot=SourceSnapshot(
            path=snapshot_path,
            dataset_id="bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
            revision="rev-bitext",
            license="CDLA-Sharing-1.0",
            sha256="0" * 64,
        ),
        output_dir=output_dir,
        lock_path=lock_path,
        split_seed=7101,
        polyai_snapshot=polyai_snapshot,
    )

    assert second_report == report
    assert report["checks"]["unresolved_placeholders"] == 0
    assert report["checks"]["cross_split_normalized_user_duplicates"] == 0
    assert report["checks"]["trainable_quarantined_records"] == 0
    assert report["checks"]["banking77_generative_sft_rows"] == 0
    assert report["checks"]["ood_duplicate_assistant_targets_allowed_only_for_task"] == "ood_gate"
    assert report["checks"]["remaining_pii_like_matches"] == 0
    assert report["checks"]["pii_like_replacements"] >= 1
    assert report["bitext_clustering"]["threshold"] > 0
    assert report["ood"]["canned_response"] == BANKING_V2_CANNED_OOD_RESPONSE
    assert report["quarantine"]["talkmap"]["trainable"] is False
    assert report["excluded_sources"]["rakesh"]["trainable"] is False
    assert report["source_roles"]["PolyAI/banking77"]["role"] == "intent-router-eval-only"
    required_patterns = {"clarification", "follow_up", "correction", "in_domain_to_ood"}
    for _split, coverage in report["synthetic_coverage"].items():
        assert coverage["ood_refusal"] >= 1
        assert set(coverage["multi_turn_patterns"]) == required_patterns

    manifest = validate_banking_v2_manifest(output_dir / "manifest.json")
    assert manifest["name"] == "hello-slm-banking-v2"
    assert [entry["name"] for entry in manifest["generative_sft"]] == [
        "train",
        "validation",
        "test",
    ]
    assert manifest["router_eval"][0]["source_dataset"] == "PolyAI/banking77"

    all_rows: list[dict] = []
    for entry in manifest["generative_sft"]:
        rows = _read_jsonl(output_dir / Path(entry["path"]).name)
        all_rows.extend(rows)
        assert entry["conversation_count"] == len(rows)
        assert entry["pii"] == "none-detected"
    router_rows = _read_jsonl(output_dir / Path(manifest["router_eval"][0]["path"]).name)
    assert router_rows
    assert all(row["metadata"]["trainable"] is False for row in router_rows)
    assert all(row["metadata"]["task"] == "intent_router_eval" for row in router_rows)

    assert all_rows
    assert len({row["conversation_id"] for row in all_rows}) == len(all_rows)
    assert any(row["metadata"]["record_type"] == "bitext_sft" for row in all_rows)
    assert any(row["metadata"]["record_type"] == "ood_refusal" for row in all_rows)
    assert any(row["metadata"]["record_type"] == "multi_turn" for row in all_rows)
    assert any(row["metadata"].get("turn_pattern") == "in_domain_to_ood" for row in all_rows)

    for row in all_rows:
        serialized = json.dumps(row)
        assert "{{" not in serialized
        assert "}}" not in serialized
        assert "4111 1111 1111 1111" not in serialized
        assert row["metadata"]["trainable"] is True
        if row["metadata"]["record_type"] == "ood_refusal":
            assert row["metadata"]["task"] == "ood_gate"
            assert row["messages"][-1]["content"] == BANKING_V2_CANNED_OOD_RESPONSE
        assert row["messages"][0]["role"] == "system"
        assert row["messages"][-1]["role"] == "assistant"

    activate_rows = [
        row
        for row in all_rows
        if row["metadata"].get("intent") == "activate_card"
        and row["metadata"]["record_type"] == "bitext_sft"
    ]
    assert len(activate_rows) == 2
    assert len({row["metadata"]["split"] for row in activate_rows}) == 1
    assert len({row["metadata"]["bitext_cluster"] for row in activate_rows}) == 1

    alternate_output_dir = tmp_path / "banking-v2-alt"
    alternate_report = prepare(
        snapshot=SourceSnapshot(
            path=snapshot_path,
            dataset_id="bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
            revision="rev-bitext",
            license="CDLA-Sharing-1.0",
            sha256="0" * 64,
        ),
        output_dir=alternate_output_dir,
        lock_path=lock_path,
        split_seed=7102,
        polyai_snapshot=polyai_snapshot,
    )
    assert (
        alternate_report["summary"]["corpus_fingerprint"]
        != report["summary"]["corpus_fingerprint"]
    )


def test_prepare_rejects_unusable_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "bad.jsonl"
    snapshot_path.write_text(
        json.dumps(
            {
                "source_row_id": 0,
                "tags": "B",
                "instruction": "",
                "category": "CARD",
                "intent": "activate_card",
                "response": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        prepare(
            snapshot=SourceSnapshot(
                path=snapshot_path,
                dataset_id="bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
                revision="rev-bitext",
                license="CDLA-Sharing-1.0",
                sha256="0" * 64,
            ),
            output_dir=tmp_path / "out",
            lock_path=tmp_path / "lock.json",
            split_seed=7101,
        )
    except BankingDataError as exc:
        assert "instruction" in str(exc)
    else:
        raise AssertionError("expected BankingDataError")
