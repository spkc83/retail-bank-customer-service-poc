from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

RUNNER_PATH = Path("scripts/banking_v2/cloud_generate_tool_eval.py")
JOB_PATH = Path("scripts/banking_v2/hf_job_tool_eval.py")
LAUNCHER_PATH = Path("scripts/banking_v2/run_remote_tool_eval_job.sh")


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module(RUNNER_PATH, "cloud_generate_tool_eval")


class TemplateTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "unit-test-granite-tool-eval-template"

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_tensors: str | None = None,
    ) -> str | dict[str, list[list[int]]]:
        assert tools is not None
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"]
        del return_tensors
        parts = []
        for message in messages:
            role = message["role"]
            if role == "assistant" and message.get("tool_calls"):
                parts.append(
                    "assistant:"
                    + json.dumps({"tool_calls": message["tool_calls"]}, sort_keys=True)
                )
            elif role == "tool":
                parts.append(
                    f"tool {message['name']}[{message['tool_call_id']}]:"
                    f"{message['content']}"
                )
            else:
                parts.append(f"{role}:{message.get('content', '')}")
        if add_generation_prompt:
            parts.append("assistant:")
        rendered = "\n".join(parts)
        if tokenize:
            return {"input_ids": [[ord(char) for char in rendered]]}
        return rendered

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        **_: Any,
    ) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, tokens: list[int], *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(int(token)) for token in tokens if int(token) > 2)


class RecordingBackend:
    tokenizer = TemplateTokenizer()

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
    ) -> str:
        assert max_new_tokens > 0
        self.calls.append([dict(message) for message in messages])
        if any(message.get("role") == "tool" for message in messages):
            return "Done. You have Main Checking ending in 1792."
        if messages[-1]["content"] == "What accounts do I have?":
            return '<tool_call>{"name":"list_accounts","arguments":{}}</tool_call>'
        return "Please provide the last four digits."


def _tool_record() -> dict[str, Any]:
    return {
        "schema_version": "banking-tool-sft/v1",
        "record_id": "tool_record",
        "messages": [
            {"role": "system", "content": "banking system", "loss": False},
            {"role": "user", "content": "What accounts do I have?", "loss": False},
            {
                "role": "assistant",
                "content": None,
                "loss": True,
                "tool_calls": [
                    {
                        "id": "call_accounts_0",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "list_accounts", "arguments": {}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_accounts_0",
                "name": "list_accounts",
                "content": {"ok": True, "result": {"accounts": [{"last4": "1792"}]}},
                "loss": False,
            },
            {
                "role": "assistant",
                "content": "Done. You have Main Checking ending in 1792.",
                "loss": True,
            },
        ],
        "expected": {
            "requires_tool": True,
            "path": "tool_success",
            "tool_calls": [{"name": "list_accounts", "arguments": {}}],
            "grounding_facts": ["account.last4=1792"],
        },
    }


def _no_tool_record() -> dict[str, Any]:
    return {
        "schema_version": "banking-tool-sft/v1",
        "record_id": "no_tool_record",
        "messages": [
            {"role": "system", "content": "banking system", "loss": False},
            {"role": "user", "content": "Please replace my card", "loss": False},
            {"role": "assistant", "content": "Please provide the last four digits.", "loss": True},
        ],
        "expected": {
            "requires_tool": False,
            "path": "clarification",
            "tool_calls": [],
            "grounding_facts": ["missing_field=last4"],
        },
    }


def _write_manifest(tmp_path: Path) -> Path:
    data_path = tmp_path / "test.jsonl"
    data_path.write_text(
        json.dumps(_tool_record(), sort_keys=True)
        + "\n"
        + json.dumps(_no_tool_record(), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"tool_sft": [{"name": "test", "path": "test.jsonl"}]}),
        encoding="utf-8",
    )
    return manifest_path


def _config(tmp_path: Path) -> Any:
    return runner.EvalConfig(
        model_repo="spkc83/retail-bank-agent-9b",
        model_revision="a" * 40,
        dataset_repo="spkc83/retail-bank-agent-sft",
        dataset_revision="b" * 40,
        manifest=_write_manifest(tmp_path),
        output_dir=tmp_path / "out",
        predictions_jsonl=None,
        metadata_json=None,
        split="test",
        family="granite",
        device="cpu",
        dtype="fp32",
        max_new_tokens_first=8,
        max_new_tokens_final=9,
        limit=None,
        trust_remote_code=False,
        push_to_hub=False,
        token=None,
    )


def test_runner_generates_two_isolated_phases_and_metadata(tmp_path: Path) -> None:
    backend = RecordingBackend()
    metadata = runner.run_eval(_config(tmp_path), backend=backend)

    predictions_path = Path(metadata["outputs"]["predictions_jsonl"])
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert [row["record_id"] for row in rows] == ["tool_record", "no_tool_record"]
    assert rows[0]["first_assistant_parsed"]["tool_calls"][0]["function"]["name"] == "list_accounts"
    assert rows[0]["grounded_final_parsed"]["content"].endswith("1792.")
    assert rows[0]["raw_output"] == (
        rows[0]["first_assistant_raw_output"]
        + "\n"
        + rows[0]["grounded_final_raw_output"]
    )
    assert rows[1]["first_assistant_parsed"]["content"] == "Please provide the last four digits."
    assert rows[1]["grounded_final_raw_output"] is None
    assert rows[1]["raw_output"] == rows[1]["first_assistant_raw_output"]
    assert [message["role"] for message in backend.calls[0]] == ["system", "user"]
    assert [message["role"] for message in backend.calls[1]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert metadata["phases"] == {
        "first_assistant_records": 2,
        "grounded_final_records": 1,
    }
    assert metadata["read_only_contract"] == {
        "tool_execution": False,
        "deterministic_output_repair": False,
        "teacher_forced_canonical_tool_results_for_grounded_final": True,
    }
    report = json.loads(
        Path(metadata["outputs"]["report_json"]).read_text(encoding="utf-8")
    )
    assert report["checkpoint_revision"] == "a" * 40
    assert report["metrics"]["tool_name_accuracy"]["score"] == 1.0
    assert report["metrics"]["grounded_final_factuality"]["score"] == 1.0


def test_runner_resumes_existing_prediction_jsonl_without_duplicates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first_metadata = runner.run_eval(config, backend=RecordingBackend())
    second_metadata = runner.run_eval(config, backend=RecordingBackend())

    rows = (
        Path(second_metadata["outputs"]["predictions_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(rows) == 2
    assert first_metadata["outputs"]["new_rows_written"] == 2
    assert second_metadata["outputs"]["new_rows_written"] == 0


def test_tool_phase_targets_tool_call_after_prior_multiturn_clarification() -> None:
    record = _tool_record()
    record["messages"][1:1] = [
        {"role": "user", "content": "I need help with an account.", "loss": False},
        {
            "role": "assistant",
            "content": "What would you like to know?",
            "loss": True,
        },
    ]

    selected = runner.first_phase_messages(record)

    assert [message["role"] for message in selected] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert selected[-1]["content"] == "What accounts do I have?"


def test_exact_revision_guard_rejects_branch_names(tmp_path: Path) -> None:
    config = runner.EvalConfig(**{**_config(tmp_path).__dict__, "model_revision": "main"})

    with pytest.raises(runner.ToolEvalGenerationError, match="exact 40-character"):
        runner.run_eval(config, backend=RecordingBackend())


def test_hf_job_requires_exact_revisions_and_invokes_eval_runner() -> None:
    job = _load_module(JOB_PATH, "hf_job_tool_eval")
    source = JOB_PATH.read_text(encoding="utf-8")

    assert "# /// script" in source
    assert '"transformers==5.13.0"' in source
    assert job.MODEL_REPO == "spkc83/retail-bank-agent-9b"
    assert job.DATASET_REPO == "spkc83/retail-bank-agent-sft"
    assert "cloud_generate_tool_eval.py" in source
    assert 'parser.add_argument("--model-repo", default=MODEL_REPO)' in source
    assert 'parser.add_argument("--dataset-repo", default=DATASET_REPO)' in source
    assert '"--model-revision",' in source
    assert '"--dataset-revision",' in source
    assert '"--push-to-hub",' in source
    with pytest.raises(ValueError, match="exact 40-character"):
        job.validate_git_revision("feat/tool-use-sft-v3", field="--model-revision")


def test_hf_eval_launcher_uses_pinned_url_durable_volume_and_two_hour_cap() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "--flavor rtx-pro-6000" in source
    assert "--timeout 2h" in source
    assert "--volume hf://buckets/spkc83/jobs-artifacts:/data" in source
    assert "curl --fail --silent --show-error --head" in source
    assert "hf_job_tool_eval.py" in source
    assert "--model-revision" in source
    assert "--dataset-revision" in source
