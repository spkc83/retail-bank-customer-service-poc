from __future__ import annotations

import json
from pathlib import Path

import pytest

from hello_slm.banking_tool_sft_data import (
    BANKING_TOOL_SFT_CONTRACT,
    BankingToolSftDataError,
    export_teacher_realization_requests,
    generate_records,
    import_teacher_realizations,
    main,
    normalized_user_text,
    prepare,
    public_tool_manifest,
    validate_banking_tool_sft_manifest,
    validate_records,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_generate_records_cover_tool_and_non_tool_contracts() -> None:
    records = generate_records(pilot_count=36, split_seed=9001)
    validate_records(records)

    assert len(records) == 36
    assert {record["schema_version"] for record in records} == {BANKING_TOOL_SFT_CONTRACT}

    manifest_names = {tool["function"]["name"] for tool in public_tool_manifest()}
    called_names = {
        call["function"]["name"]
        for record in records
        for message in record["messages"]
        for call in message.get("tool_calls", [])
    }
    assert called_names == manifest_names

    tool_messages = [
        message
        for record in records
        for message in record["messages"]
        if message["role"] == "tool"
    ]
    assert any(
        message["content"]["ok"] is True and "result" in message["content"]
        for message in tool_messages
    )
    assert any(
        message["content"]["ok"] is False and "error" in message["content"]
        for message in tool_messages
    )

    cases = {record["expected"]["path"] for record in records}
    assert {
        "tool_success",
        "tool_error",
        "clarification",
        "no_tool_banking_faq",
        "ood",
        "hard_negative",
        "multi_turn",
    } <= cases

    assert any(len(record["expected"]["ordered_calls"]) >= 2 for record in records)
    assert any(
        len([m for m in record["messages"] if m["role"] == "user"]) > 1
        for record in records
    )


def test_tool_calls_have_stable_ids_typed_args_and_replay_hashes() -> None:
    first = generate_records(pilot_count=18, split_seed=711)
    second = generate_records(pilot_count=18, split_seed=711)

    assert second == first
    for record in first:
        ordered_calls = record["expected"]["ordered_calls"]
        call_ids = [
            call["id"]
            for message in record["messages"]
            for call in message.get("tool_calls", [])
        ]
        assert call_ids == ordered_calls
        for index, call_id in enumerate(call_ids):
            assert call_id == f"call_{record['record_id']}_{index}"
        for message in record["messages"]:
            for call in message.get("tool_calls", []):
                assert isinstance(call["function"]["arguments"], dict)
                assert call["index"] == ordered_calls.index(call["id"])
        assert record["validation"]["accepted"] is True
        assert record["validation"]["tool_manifest_hash"].startswith("sha256:")
        assert record["validation"]["replay_hash"].startswith("sha256:")
        if ordered_calls:
            assert record["expected"]["final_state_hash"].startswith("sha256:")


def test_prepare_writes_manifest_report_and_is_split_isolated(tmp_path: Path) -> None:
    report = prepare(output_dir=tmp_path / "tool-sft", pilot_count=120, split_seed=1234)
    second_report = prepare(output_dir=tmp_path / "tool-sft", pilot_count=120, split_seed=1234)

    assert second_report == report
    assert report["summary"]["total_records"] == 120
    assert report["checks"]["accepted_records"] == 120
    assert report["checks"]["tool_names_covered"] == sorted(
        tool["function"]["name"] for tool in public_tool_manifest()
    )
    assert report["checks"]["banking77_generative_sft_rows"] == 0
    assert report["checks"]["bitext_rows"] == 0
    assert report["quarantine"]["bitext"]["trainable"] is False
    data_card = (tmp_path / "tool-sft" / "README.md").read_text(encoding="utf-8")
    assert "Retail Bank Agent Tool-Use SFT" in data_card
    assert "120 deterministic" in data_card
    assert "Banking77" in data_card
    assert "classifier/evaluation-only" in data_card

    manifest = validate_banking_tool_sft_manifest(tmp_path / "tool-sft" / "manifest.json")
    assert manifest["contract"] == "banking-tool-sft-manifest"
    assert [entry["name"] for entry in manifest["tool_sft"]] == [
        "train",
        "validation",
        "test",
    ]

    semantic_split_by_group: dict[tuple[str, str, str, str], str] = {}
    for entry in manifest["tool_sft"]:
        assert entry["path"] == f"{entry['name']}.jsonl"
        rows = _read_jsonl(tmp_path / "tool-sft" / entry["path"])
        assert rows
        assert entry["record_count"] == len(rows)
        for row in rows:
            semantic_group = tuple(
                row["split_keys"][key]
                for key in (
                    "scenario_family",
                    "state_seed",
                    "customer_id",
                    "template_id",
                )
            )
            assert row["metadata"]["split_group"] == "|".join(semantic_group)
            assigned_split = semantic_split_by_group.setdefault(semantic_group, entry["name"])
            assert assigned_split == entry["name"]

    realized_groups = {
        tuple(
            row["split_keys"][key]
            for key in (
                "scenario_family",
                "state_seed",
                "customer_id",
                "template_id",
            )
        )
        for entry in manifest["tool_sft"]
        for row in _read_jsonl(tmp_path / "tool-sft" / entry["path"])
        if row["split_keys"]["realization_seed"] != "realization-000"
    }
    assert realized_groups


def test_pilot_realizer_uses_natural_text_and_varied_state_slots() -> None:
    records = generate_records(pilot_count=1200, split_seed=4321)
    user_keys = [
        normalized_user_text(
            next(
                message["content"]
                for message in reversed(record["messages"])
                if message["role"] == "user"
            )
        )
        for record in records
    ]

    assert len(user_keys) == 1200
    assert len(set(user_keys)) == 1200
    serialized_users = "\n".join(
        message["content"]
        for record in records
        for message in record["messages"]
        if message["role"] == "user"
    )
    assert "Use the " not in serialized_users
    assert " phrasing from the " not in serialized_users

    assert len({record["split_keys"]["state_seed"] for record in records}) >= 100
    write_values: dict[tuple[str, str], set[str]] = {
        ("freeze_card", "last4"): set(),
        ("replace_card", "last4"): set(),
        ("dispute_transaction", "description"): set(),
        ("cancel_transfer", "recipient"): set(),
    }
    for record in records:
        for message in record["messages"]:
            for call in message.get("tool_calls", []):
                name = call["function"]["name"]
                arguments = call["function"]["arguments"]
                for key in list(write_values):
                    if key[0] == name and key[1] in arguments:
                        write_values[key].add(arguments[key[1]])

    assert all(len(values) >= 20 for values in write_values.values())

    final_responses = [
        str(record["messages"][-1]["content"]).strip() for record in records
    ]
    assert all(len(normalized_user_text(response).split()) >= 7 for response in final_responses)
    by_path = {
        path: [
            response
            for record, response in zip(records, final_responses, strict=True)
            if record["expected"]["path"] == path
        ]
        for path in {
            "clarification",
            "no_tool_banking_faq",
            "ood",
            "hard_negative",
        }
    }
    assert all("last four digits" in response.lower() for response in by_path["clarification"])
    assert all("overdraft" in response.lower() for response in by_path["no_tool_banking_faq"])
    assert all("retail banking" in response.lower() for response in by_path["ood"])
    assert all(
        "account number" in response.lower() and "customer id" in response.lower()
        for response in by_path["hard_negative"]
    )


def test_teacher_realization_round_trip_allows_only_wording_changes(tmp_path: Path) -> None:
    records = generate_records(pilot_count=18, split_seed=711)
    request_path = tmp_path / "teacher-requests.jsonl"
    response_path = tmp_path / "teacher-responses.jsonl"

    export_teacher_realization_requests(records, request_path)
    rows = _read_jsonl(request_path)
    rows[0]["user_content"] = "Please check the accounts on this signed-in profile."
    rows[0]["final_response"] = "I checked your profile and found your account summary."
    with response_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    realized = import_teacher_realizations(
        records,
        response_path,
        teacher_model="teacher-test",
        teacher_prompt_hash="sha256:" + "1" * 64,
    )

    assert realized[0]["messages"][1]["content"] == rows[0]["user_content"]
    assert realized[0]["messages"][-1]["content"] == rows[0]["final_response"]
    assert realized[0]["messages"][2:] != []
    assert realized[0]["expected"] == records[0]["expected"]
    assert realized[0]["provenance"]["teacher_model"] == "teacher-test"
    validate_records(realized)

    rows[0]["immutable_hash"] = "sha256:" + "0" * 64
    with response_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with pytest.raises(BankingToolSftDataError, match="teacher request hash mismatch"):
        import_teacher_realizations(
            records,
            response_path,
            teacher_model="teacher-test",
            teacher_prompt_hash="sha256:" + "1" * 64,
        )


def test_cli_exports_and_applies_teacher_realizations(tmp_path: Path) -> None:
    output_dir = tmp_path / "tool-sft"
    request_path = tmp_path / "teacher-requests.jsonl"
    response_path = tmp_path / "teacher-responses.jsonl"

    assert (
        main(
            [
                "--output-dir",
                str(output_dir),
                "--pilot-count",
                "18",
                "--export-teacher-requests",
                str(request_path),
            ]
        )
        == 0
    )
    rows = _read_jsonl(request_path)
    rows[0]["user_content"] = "Please summarize the accounts for this signed-in profile."
    rows[0]["final_response"] = "I checked the signed-in profile and found the account summary."
    with response_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    assert (
        main(
            [
                "--output-dir",
                str(output_dir),
                "--pilot-count",
                "18",
                "--teacher-responses",
                str(response_path),
                "--teacher-model",
                "teacher-cli-test",
                "--teacher-prompt-hash",
                "sha256:" + "2" * 64,
            ]
        )
        == 0
    )
    manifest = validate_banking_tool_sft_manifest(output_dir / "manifest.json")
    split_paths = [entry["path"] for entry in manifest["tool_sft"]]
    assert split_paths == ["train.jsonl", "validation.jsonl", "test.jsonl"]
    all_rows = [
        row
        for entry in manifest["tool_sft"]
        for row in _read_jsonl(output_dir / entry["path"])
    ]
    realized = next(row for row in all_rows if row["record_id"] == rows[0]["record_id"])
    assert realized["provenance"]["teacher_model"] == "teacher-cli-test"
    assert realized["messages"][-1]["content"] == rows[0]["final_response"]


def test_validator_rejects_private_or_unknown_tool_arguments() -> None:
    record = generate_records(pilot_count=18)[0]
    assistant = next(message for message in record["messages"] if message.get("tool_calls"))
    assistant["tool_calls"][0]["function"]["arguments"]["customer_id"] = "cust_alex"

    with pytest.raises(BankingToolSftDataError, match="unsupported arguments"):
        validate_records([record])


def test_validator_rejects_semantically_empty_final_response() -> None:
    record = generate_records(pilot_count=18)[0]
    record["messages"][-1]["content"] = "Done."

    with pytest.raises(BankingToolSftDataError, match="missing semantic content"):
        validate_records([record])
