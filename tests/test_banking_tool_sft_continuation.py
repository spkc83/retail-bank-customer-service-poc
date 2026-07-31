from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

WORKER_PATH = Path("scripts/retail_bank/cloud_continue_tool_sft.py")
JOB_PATH = Path("scripts/retail_bank/hf_job_continue_tool_sft.py")
LAUNCHER_PATH = Path("scripts/retail_bank/run_remote_continuation_job.sh")


def _load_worker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cloud_continue_tool_sft", WORKER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WORKER = _load_worker()


def _record(
    record_id: str,
    *,
    path: str,
    assistant_tool_calls: int = 0,
    final: str = "Done.",
    requires_tool: bool = False,
) -> dict[str, object]:
    tool_calls = [
        {
            "id": f"call_{record_id}_{index}",
            "index": index,
            "type": "function",
            "function": {"name": "list_cards", "arguments": {}},
        }
        for index in range(assistant_tool_calls)
    ]
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "demo", "loss": False},
        {"role": "user", "content": "help", "loss": False},
    ]
    if tool_calls:
        messages.append(
            {"role": "assistant", "content": None, "loss": True, "tool_calls": tool_calls}
        )
        for call in tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "name": "list_cards",
                    "tool_call_id": call["id"],
                    "content": {"ok": True, "result": {}},
                    "loss": False,
                }
            )
    messages.append({"role": "assistant", "content": final, "loss": True})
    return {
        "record_id": record_id,
        "messages": messages,
        "expected": {"path": path, "requires_tool": requires_tool},
        "metadata": {"scenario_family": record_id},
    }


def test_continuation_requires_exact_model_revision() -> None:
    with pytest.raises(ValueError, match="exact 40-character"):
        WORKER.require_exact_revision("main", field="--source-model-revision")

    WORKER.require_exact_revision("00c4ba1be926fc26dbc1f5311a4fd037462be1c1", field="ok")


def test_continuation_mix_oversamples_sequential_and_safe_clarification() -> None:
    sequential = _record(
        "sequential",
        path="multi_turn",
        assistant_tool_calls=2,
        final="I found the active card and froze it.",
        requires_tool=True,
    )
    clarification = _record(
        "clarification",
        path="clarification",
        final="Which card should I replace? Please provide the last four digits shown in the app.",
    )
    single_tool = _record(
        "single",
        path="tool_success",
        assistant_tool_calls=1,
        final="Your balance is ready.",
        requires_tool=True,
    )
    faq = _record(
        "faq",
        path="no_tool_banking_faq",
        final="Overdraft fees depend on account disclosures.",
    )

    mixed, stats = WORKER.build_continuation_mix(
        [sequential, clarification, single_tool, faq],
        sequential_multiplier=5,
        clarification_multiplier=4,
        servicing_quality_multiplier=4,
        seed=123,
    )
    counts = {
        "sequential": sum(1 for record in mixed if record["record_id"] == "sequential"),
        "clarification": sum(1 for record in mixed if record["record_id"] == "clarification"),
        "single": sum(1 for record in mixed if record["record_id"] == "single"),
        "faq": sum(1 for record in mixed if record["record_id"] == "faq"),
    }

    assert counts == {"sequential": 5, "clarification": 4, "single": 1, "faq": 1}
    assert stats["sequential_focus_records"] == 1
    assert stats["credential_safe_clarification_records"] == 1
    assert stats["regression_records"] == 2


def test_continuation_mix_oversamples_servicing_quality_families() -> None:
    balances = _record(
        "read_accounts",
        path="tool_success",
        assistant_tool_calls=1,
        final="Everyday Checking has USD 3,245.67 available.",
        requires_tool=True,
    )
    mortgage_age = _record(
        "faq_mortgage_age",
        path="no_tool_banking_faq",
        final="Applicants are typically at least 18.",
    )
    unrelated = _record("write_card", path="tool_success", assistant_tool_calls=1)

    mixed, stats = WORKER.build_continuation_mix(
        [balances, mortgage_age, unrelated],
        sequential_multiplier=5,
        clarification_multiplier=4,
        servicing_quality_multiplier=4,
        seed=123,
    )
    counts = {
        record_id: sum(1 for record in mixed if record["record_id"] == record_id)
        for record_id in ("read_accounts", "faq_mortgage_age", "write_card")
    }

    assert counts == {"read_accounts": 4, "faq_mortgage_age": 4, "write_card": 1}
    assert stats["servicing_quality_records"] == 2
    assert stats["servicing_quality_multiplier"] == 4


def test_unsafe_clarification_is_not_focus_oversampled() -> None:
    unsafe = _record(
        "unsafe",
        path="clarification",
        final="Please provide your password and customer ID.",
    )

    mixed, stats = WORKER.build_continuation_mix(
        [unsafe],
        sequential_multiplier=5,
        clarification_multiplier=4,
        servicing_quality_multiplier=4,
    )

    assert len(mixed) == 1
    assert stats["credential_safe_clarification_records"] == 0


def test_worker_dry_run_exposes_capped_continuation_plan() -> None:
    config = WORKER.config_from_args(WORKER.parse_args([]))
    plan = WORKER.build_dry_run_plan(config)

    assert plan["worker"] == "cloud_continue_tool_sft"
    assert plan["source_model_revision"] == "00c4ba1be926fc26dbc1f5311a4fd037462be1c1"
    assert plan["training"]["max_steps"] == 600
    assert plan["training"]["max_train_seconds"] == 9_000
    assert plan["training"]["servicing_quality_multiplier"] == 4
    assert plan["release"]["merge"] == "existing FP32 accumulation, FP16 saved weights"
    assert plan["remote_guard"]["currently_allowed"] is False


def test_continuation_job_bootstrap_is_pinned_to_worker_and_dependencies() -> None:
    source = JOB_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }

    assert "# /// script" in source
    assert '"trl==0.26.2"' in source
    assert assignments["MODEL_REPO"] == "spkc83/retail-bank-agent-9b"
    assert assignments["BASE_REVISION"] == "1504002f650e656a0a3789d99574df12e3e94ed0"
    assert "cloud_continue_tool_sft.py" in source
    assert "cloud_train_tool_sft.py" not in source
    assert "--source-model-revision" in source
    assert "RETAIL_BANK_ALLOW_REMOTE_CONTINUATION_SFT" in source


def test_remote_continuation_launcher_mounts_durable_bucket_and_uses_five_hour_cap() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "--timeout 5h" in launcher
    assert "--volume hf://buckets/spkc83/jobs-artifacts:/data" in launcher
    assert "SOURCE_MODEL_REVISION must be the exact 40-character lowercase Git commit" in launcher
    assert (
        "retail-bank-agent-9b-continuation-${source_commit:0:8}-"
        "${source_model_revision:0:8}"
    ) in launcher
    assert "hf jobs uv run" in launcher
    assert "/scripts/retail_bank/hf_job_continue_tool_sft.py" in launcher
    assert "/scripts/banking_v2/hf_job_continue_tool_sft.py" not in launcher
    assert 'script_url="$legacy_script_url"' not in launcher
    assert "rm " not in launcher


def test_worker_release_runs_gates_before_upload() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    remote_body = source.split("def run_remote_continuation", 1)[1].split(
        "def main", 1
    )[0]

    assert remote_body.index("release = run_release_tools(config)") < remote_body.index(
        "if config.push_to_hub:"
    )
    assert "validate_parity(" in source
    assert "hf_job_remerge_tool_sft.py" in source
    assert "hf_job_merge_parity.py" in source


def test_worker_enables_input_grads_for_trainable_peft_checkpointing() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    remote_body = source.split("def run_remote_continuation", 1)[1].split(
        "train_output = trainer.train", 1
    )[0]

    assert remote_body.index("PeftModel.from_pretrained(") < remote_body.index(
        "model.enable_input_require_grads()"
    )
    assert remote_body.index("model.enable_input_require_grads()") < remote_body.index(
        "trainer = SFTTrainer("
    )


def test_continuation_upload_replaces_release_evidence_files() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    upload_body = source.split("def upload_release", 1)[1].split(
        "def run_remote_continuation", 1
    )[0]

    assert 'path_in_repo="merge_parity_diagnostics.json"' in upload_body
    assert 'path_in_repo="fp16_remerge.json"' in upload_body
    assert "merge_parity_diagnostics_merged-fp16_float16.json" in upload_body
