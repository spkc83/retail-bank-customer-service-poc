from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

JOB_PATH = Path("scripts/retail_bank/hf_job_tool_sft.py")


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
    assert 'parser.add_argument("--resume-from")' in source
    assert 'command.extend(["--resume-from", args.resume_from])' in source


def test_remote_launcher_mounts_durable_job_bucket() -> None:
    launcher = Path(
        "scripts/retail_bank/run_remote_training_job.sh"
    ).read_text(encoding="utf-8")

    assert "--volume hf://buckets/spkc83/jobs-artifacts:/data" in launcher
    assert '--output-dir "/data/retail-bank-agent-9b-${source_commit:0:8}"' in launcher
    assert "must be the exact 40-character lowercase Git commit" in launcher
    assert "/scripts/retail_bank/hf_job_tool_sft.py" in launcher
    assert "/scripts/banking_v2/hf_job_tool_sft.py" in launcher
    assert 'if ! curl --fail --silent --head "$script_url"' in launcher
    assert 'script_url="$legacy_script_url"' in launcher
    assert 'job_args+=(--resume-from "$resume_from")' in launcher


@pytest.mark.parametrize(
    ("launcher", "arguments", "bootstrap"),
    [
        (
            "run_remote_training_job.sh",
            ("a" * 40, "b" * 40),
            "hf_job_tool_sft.py",
        ),
        (
            "run_remote_continuation_job.sh",
            ("a" * 40, "b" * 40, "c" * 40),
            "hf_job_continue_tool_sft.py",
        ),
        (
            "run_remote_continuation_export_recovery.sh",
            (
                "a" * 40,
                "b" * 40,
                "c" * 40,
                "d" * 40,
                "spkc83/job-123",
                "/data/retail-bank-agent-9b-continuation-test",
                "600",
            ),
            "hf_job_recover_continuation_export.py",
        ),
        (
            "run_remote_tool_eval_job.sh",
            ("a" * 40, "b" * 40, "c" * 40),
            "hf_job_tool_eval.py",
        ),
    ],
)
@pytest.mark.parametrize("available_path", ["current", "legacy"])
def test_remote_launchers_resolve_current_and_pre_rename_bootstraps(
    tmp_path: Path,
    launcher: str,
    arguments: tuple[str, ...],
    bootstrap: str,
    available_path: str,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl_log = tmp_path / "curl.log"
    hf_log = tmp_path / "hf.log"
    curl = bin_dir / "curl"
    curl.write_text(
        """#!/usr/bin/env bash
url="${@: -1}"
printf '%s\\n' "$url" >> "$CURL_LOG"
if [[ "$MOCK_CURL_PATH" == "current" ]]; then
  [[ "$url" == *"/scripts/retail_bank/"* ]]
else
  [[ "$url" == *"/scripts/banking_v2/"* ]]
fi
""",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    hf = bin_dir / "hf"
    hf.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$HF_LOG"
""",
        encoding="utf-8",
    )
    hf.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CURL_LOG": str(curl_log),
        "HF_LOG": str(hf_log),
        "MOCK_CURL_PATH": available_path,
    }

    subprocess.run(
        ["bash", f"scripts/retail_bank/{launcher}", *arguments],
        check=True,
        env=env,
    )

    requested_urls = curl_log.read_text(encoding="utf-8").splitlines()
    submitted_args = hf_log.read_text(encoding="utf-8")
    expected_segment = (
        "/scripts/retail_bank/"
        if available_path == "current"
        else "/scripts/banking_v2/"
    )
    assert requested_urls[0].endswith(f"/scripts/retail_bank/{bootstrap}")
    assert len(requested_urls) == (1 if available_path == "current" else 2)
    assert f"{expected_segment}{bootstrap}" in submitted_args


def test_post_training_evaluation_detaches_closed_trackio_callback() -> None:
    worker_source = Path(
        "scripts/retail_bank/cloud_train_tool_sft.py"
    ).read_text(encoding="utf-8")

    assert "trainer.remove_callback(TrackioCallback)" in worker_source
    assert worker_source.index("trainer.remove_callback(TrackioCallback)") < (
        worker_source.index("eval_metrics = trainer.evaluate()")
    )
