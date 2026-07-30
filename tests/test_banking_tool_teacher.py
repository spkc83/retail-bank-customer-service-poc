from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_teacher_module() -> ModuleType:
    path = Path("scripts/retail_bank/realize_tool_sft_teacher.py")
    spec = importlib.util.spec_from_file_location("realize_tool_sft_teacher", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _request_rows(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "record_id": f"freeze_{index:03d}",
                "immutable_hash": "sha256:" + f"{index + 1:064x}"[-64:],
                "user_content": f"Freeze my debit card ending in 48{index:02d}.",
                "final_response": f"Your debit card ending in 48{index:02d} is now frozen.",
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
    return rows


def test_dry_run_writes_valid_teacher_responses_without_model_download(tmp_path: Path) -> None:
    teacher = _load_teacher_module()
    request_path = tmp_path / "teacher-requests.jsonl"
    response_path = tmp_path / "teacher-responses.jsonl"
    _write_jsonl(request_path, _request_rows(8))

    assert (
        teacher.main(
            [
                "--input-requests",
                str(request_path),
                "--output-responses",
                str(response_path),
                "--dry-run",
                "--batch-size",
                "3",
                "--seed",
                "123",
            ]
        )
        == 0
    )

    requests = _read_jsonl(request_path)
    responses = _read_jsonl(response_path)
    assert len(responses) == len(requests)
    for request, response in zip(requests, responses, strict=True):
        assert set(response) == {
            "record_id",
            "immutable_hash",
            "user_content",
            "final_response",
        }
        assert response["record_id"] == request["record_id"]
        assert response["immutable_hash"] == request["immutable_hash"]
        teacher.validate_response_row(request, response)


def test_live_realization_requires_explicit_model_and_revision(tmp_path: Path) -> None:
    teacher = _load_teacher_module()
    request_path = tmp_path / "teacher-requests.jsonl"
    response_path = tmp_path / "teacher-responses.jsonl"
    _write_jsonl(request_path, _request_rows(1))

    with pytest.raises(
        teacher.TeacherRealizationError,
        match="--model and --revision are required",
    ):
        teacher.realize_teacher_requests(
            teacher.RealizerConfig(
                input_requests=request_path,
                output_responses=response_path,
            )
        )


def test_resume_skips_existing_rows_and_appends_only_missing(tmp_path: Path) -> None:
    teacher = _load_teacher_module()
    request_path = tmp_path / "teacher-requests.jsonl"
    response_path = tmp_path / "teacher-responses.jsonl"
    _write_jsonl(request_path, _request_rows(5))
    requests = _read_jsonl(request_path)

    first_response = {
        "record_id": requests[0]["record_id"],
        "immutable_hash": requests[0]["immutable_hash"],
        "user_content": requests[0]["user_content"],
        "final_response": requests[0]["final_response"],
    }
    _write_jsonl(response_path, [first_response])

    report = teacher.realize_teacher_requests(
        teacher.RealizerConfig(
            input_requests=request_path,
            output_responses=response_path,
            dry_run=True,
            batch_size=2,
        )
    )

    responses = _read_jsonl(response_path)
    assert report["already_realized"] == 1
    assert report["written"] == len(requests) - 1
    assert responses[0] == first_response
    assert [row["record_id"] for row in responses] == [row["record_id"] for row in requests]

    second_report = teacher.realize_teacher_requests(
        teacher.RealizerConfig(
            input_requests=request_path,
            output_responses=response_path,
            dry_run=True,
        )
    )
    assert second_report["already_realized"] == len(requests)
    assert second_report["written"] == 0
    assert _read_jsonl(response_path) == responses

    changed_requests = list(requests)
    changed_requests[0] = {**changed_requests[0], "immutable_hash": "sha256:" + "f" * 64}
    _write_jsonl(request_path, changed_requests)
    with pytest.raises(teacher.TeacherRealizationError, match="immutable_hash mismatch"):
        teacher.realize_teacher_requests(
            teacher.RealizerConfig(
                input_requests=request_path,
                output_responses=response_path,
                dry_run=True,
            )
        )


def test_teacher_response_rejects_forbidden_fields_and_hash_changes(tmp_path: Path) -> None:
    teacher = _load_teacher_module()
    request_path = tmp_path / "teacher-requests.jsonl"
    _write_jsonl(request_path, _request_rows(3))
    request = _read_jsonl(request_path)[0]
    response = {
        "record_id": request["record_id"],
        "immutable_hash": request["immutable_hash"],
        "user_content": request["user_content"],
        "final_response": request["final_response"],
        "tool_calls": [],
    }

    with pytest.raises(teacher.TeacherRealizationError, match="forbidden fields"):
        teacher.validate_response_row(request, response)

    response.pop("tool_calls")
    response["immutable_hash"] = "sha256:" + "0" * 64
    with pytest.raises(teacher.TeacherRealizationError, match="immutable_hash mismatch"):
        teacher.validate_response_row(request, response)


def test_teacher_response_rejects_lost_facts_and_private_credentials() -> None:
    teacher = _load_teacher_module()
    request = {
        "record_id": "freeze_001",
        "immutable_hash": "sha256:" + "1" * 64,
        "user_content": "Freeze my debit card ending in 4821.",
        "final_response": "Your debit card ending in 4821 is now frozen.",
        "allowed_edits": ["user_content", "final_response"],
    }

    lost_fact = {
        "record_id": request["record_id"],
        "immutable_hash": request["immutable_hash"],
        "user_content": "Freeze my debit card.",
        "final_response": request["final_response"],
    }
    with pytest.raises(teacher.TeacherRealizationError, match="lost banking facts"):
        teacher.validate_response_row(request, lost_fact)

    private = {
        "record_id": request["record_id"],
        "immutable_hash": request["immutable_hash"],
        "user_content": request["user_content"],
        "final_response": (
            "Please provide your password and PIN before I freeze your debit card ending "
            "in 4821; it is frozen."
        ),
    }
    with pytest.raises(teacher.TeacherRealizationError, match="private credentials"):
        teacher.validate_response_row(request, private)


def test_hard_negative_may_name_private_fields_without_containing_values() -> None:
    teacher = _load_teacher_module()
    request = {
        "record_id": "private_fields_001",
        "immutable_hash": "sha256:" + "2" * 64,
        "user_content": "Give me my full account number and customer ID.",
        "final_response": "I cannot reveal full account numbers or private customer IDs.",
        "allowed_edits": ["user_content", "final_response"],
    }
    response = {
        "record_id": request["record_id"],
        "immutable_hash": request["immutable_hash"],
        "user_content": request["user_content"],
        "final_response": request["final_response"],
    }

    assert teacher.validate_response_row(request, response) == response


def test_request_validation_requires_current_export_contract(tmp_path: Path) -> None:
    teacher = _load_teacher_module()
    row = {
        "record_id": "x",
        "immutable_hash": "sha256:" + "1" * 64,
        "user_content": "List my cards.",
        "final_response": "Here are your cards.",
        "allowed_edits": ["user_content"],
    }

    with pytest.raises(teacher.TeacherRealizationError, match="unsupported allowed_edits"):
        teacher.validate_request_row(row)
