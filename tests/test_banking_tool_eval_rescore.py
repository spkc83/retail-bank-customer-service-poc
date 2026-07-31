from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_module() -> ModuleType:
    path = Path("scripts/retail_bank/rescore_tool_eval.py")
    spec = importlib.util.spec_from_file_location("rescore_tool_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rescore = _load_module()


def _record(record_id: str, *, expected_value: str = "old") -> dict[str, Any]:
    return {
        "record_id": record_id,
        "messages": [
            {"role": "system", "content": "banking"},
            {"role": "user", "content": "When was it created?"},
        ],
        "expected": {"grounding_facts": [expected_value]},
    }


def test_prompt_equivalence_allows_only_expected_metadata_to_change() -> None:
    generation = [_record("one", expected_value="old")]
    scoring = [_record("one", expected_value="corrected")]

    evidence = rescore.verify_prompt_equivalence(generation, scoring)

    assert evidence["record_count"] == 1
    assert evidence["messages_sha256"].startswith("sha256:")
    assert evidence["changed_expected_records"] == 1


def test_prompt_equivalence_rejects_message_or_order_drift() -> None:
    generation = [_record("one"), _record("two")]
    changed_message = [_record("one"), _record("two")]
    changed_message[1]["messages"][-1]["content"] = "Different prompt"

    with pytest.raises(rescore.RescoreError, match="messages differ for record two"):
        rescore.verify_prompt_equivalence(generation, changed_message)
    with pytest.raises(rescore.RescoreError, match="record IDs or ordering differ"):
        rescore.verify_prompt_equivalence(generation, list(reversed(generation)))


def test_prediction_coverage_requires_exact_record_set(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"record_id": "one", "raw_output": "answer"}) + "\n",
        encoding="utf-8",
    )

    outputs = rescore.load_exact_predictions(predictions, [_record("one")])
    assert outputs == {"one": "answer"}

    with pytest.raises(rescore.RescoreError, match="prediction coverage mismatch"):
        rescore.load_exact_predictions(predictions, [_record("one"), _record("two")])
