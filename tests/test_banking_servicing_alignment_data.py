from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

from hello_slm.banking_servicing_alignment_data import (
    SCREENSHOT_HELDOUT_CURRENTS,
    build_servicing_alignment_splits,
    validate_servicing_alignment_splits,
    write_servicing_alignment_dataset,
)
from hello_slm.banking_tool_sft_data import validate_banking_tool_sft_manifest


def _last_user(record: dict[str, Any]) -> str:
    for message in reversed(record["messages"]):
        if message["role"] == "user":
            return str(message["content"])
    raise AssertionError("missing user message")


def _normalize(text: str) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in text).split()
    )


def _load_preparation_module() -> ModuleType:
    path = Path("scripts/retail_bank/prepare_servicing_alignment_data.py")
    spec = importlib.util.spec_from_file_location(
        "prepare_servicing_alignment_data",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_servicing_alignment_records_validate_and_cover_failure_modes() -> None:
    splits, report = build_servicing_alignment_splits()

    validate_servicing_alignment_splits(splits)
    assert report["split_counts"] == {
        "train": 320,
        "validation": 80,
        "test": 27,
    }
    train_families = Counter(
        record["metadata"]["scenario_family"] for record in splits["train"]
    )
    assert train_families == {
        "service_case_context": 64,
        "card_anaphora_action": 64,
        "clarification_answer": 64,
        "agent_repair": 64,
        "external_topic_shift": 32,
        "banking_topic_shift": 32,
    }
    service_case_records = [
        record
        for record in splits["train"]
        if record["metadata"]["scenario_family"] == "service_case_context"
    ]
    assert service_case_records
    for record in service_case_records:
        assert record["expected"]["tool_calls"] == [
            {"name": "list_service_cases", "arguments": {}}
        ]
        final_text = record["messages"][-1]["content"]
        assert "2026-06-18" in final_text
        assert "address_update" in final_text
        assert "Confirm mailing address update" in final_text

    created_at_records = [
        record
        for record in service_case_records
        if str(record["record_id"]).startswith("svc_case_created_")
    ]
    assert created_at_records
    assert all(
        record["expected"]["grounding_facts"]
        == ["case.created_at=2026-06-18T14:00:00Z"]
        for record in created_at_records
    )
    ood_records = [
        record
        for records in splits.values()
        for record in records
        if record["expected"]["path"] == "ood"
    ]
    assert ood_records
    assert all(record["expected"]["grounding_facts"] == [] for record in ood_records)


def test_exact_screenshot_currents_are_held_out_from_training() -> None:
    splits, _report = build_servicing_alignment_splits()
    heldout = {_normalize(text) for text in SCREENSHOT_HELDOUT_CURRENTS}
    train_currents = {_normalize(_last_user(record)) for record in splits["train"]}
    test_currents = {_normalize(_last_user(record)) for record in splits["test"]}

    assert not train_currents & heldout
    assert {
        "when was that created",
        "ok thats the one i want to replace",
        "what about the weather there",
    } <= test_currents


def test_writer_outputs_manifest_and_schema_valid_splits(tmp_path: Path) -> None:
    manifest = write_servicing_alignment_dataset(tmp_path)

    assert manifest["name"] == "retail-bank-servicing-alignment-v4"
    assert manifest["schema_version"] == "banking-tool-sft/v1"
    assert manifest["report"]["alignment_split_counts"] == {
        "train": 320,
        "validation": 80,
        "test": 27,
    }
    assert manifest["report"]["base_split_counts"] == {
        "train": 6304,
        "validation": 1349,
        "test": 1347,
    }
    assert manifest["report"]["split_counts"] == {
        "train": 6624,
        "validation": 1429,
        "test": 1374,
    }
    assert {entry["name"] for entry in manifest["tool_sft"]} == {
        "train",
        "validation",
        "test",
    }
    for entry in manifest["tool_sft"]:
        path = tmp_path / entry["path"]
        assert path.is_file()
        assert path.stat().st_size == entry["bytes"]
        assert len(path.read_text(encoding="utf-8").splitlines()) == entry["record_count"]
    validate_banking_tool_sft_manifest(tmp_path / "manifest.json")

    report = json.loads((tmp_path / "preparation-report.json").read_text())
    assert report["pii_matches"] == 0
    assert report["heldout_exact_currents_in_train"] == []


def test_release_lock_detects_split_drift(tmp_path: Path) -> None:
    preparation = _load_preparation_module()
    manifest = write_servicing_alignment_dataset(tmp_path / "dataset")
    preparation.verify_release_lock(
        manifest,
        Path("data/sources/banking-servicing-alignment-v4.lock.json"),
    )
    lock = json.loads(
        Path("data/sources/banking-servicing-alignment-v4.lock.json").read_text()
    )
    lock["prepared_split_sha256"]["train"] = "0" * 64
    bad_lock = tmp_path / "bad.lock.json"
    bad_lock.write_text(json.dumps(lock), encoding="utf-8")

    try:
        preparation.verify_release_lock(manifest, bad_lock)
    except ValueError as error:
        assert "split digests drifted" in str(error)
    else:
        raise AssertionError("drifted lock should fail")
