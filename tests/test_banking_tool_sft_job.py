from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

JOB_PATH = Path("scripts/banking_v2/hf_job_tool_sft.py")


def _load_job() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hf_job_tool_sft", JOB_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_job_script_has_inline_dependencies_and_pinned_artifacts() -> None:
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
    assert '"trackio>=0.33,<0.34"' in source
    assert assignments["MODEL_REPO"] == "spkc83/retail-bank-agent-9b"
    assert assignments["DATASET_REPO"] == "spkc83/retail-bank-agent-sft"
    assert assignments["BASE_REVISION"] == "1504002f650e656a0a3789d99574df12e3e94ed0"


def test_source_download_requires_a_commit_hash_before_network(
    tmp_path: Path,
) -> None:
    job = _load_job()

    with pytest.raises(ValueError, match="Git commit"):
        job.download_source("feat/tool-use-sft-v3", tmp_path)


def test_job_command_preserves_five_hour_internal_budget() -> None:
    source = JOB_PATH.read_text(encoding="utf-8")

    assert 'default=14_400' in source
    assert 'default="/data/retail-bank-agent-9b"' in source
    assert "snapshot_download(" in source
    assert 'repo_type="dataset"' in source
    assert "dataset manifest is unavailable" in source
    assert '"--precision",' in source
    assert '"bf16-lora",' in source
    assert '"--push-to-hub",' in source


def test_post_training_evaluation_detaches_closed_trackio_callback() -> None:
    worker_source = Path(
        "scripts/banking_v2/cloud_train_tool_sft.py"
    ).read_text(encoding="utf-8")

    assert "trainer.remove_callback(TrackioCallback)" in worker_source
    assert worker_source.index("trainer.remove_callback(TrackioCallback)") < (
        worker_source.index("eval_metrics = trainer.evaluate()")
    )
