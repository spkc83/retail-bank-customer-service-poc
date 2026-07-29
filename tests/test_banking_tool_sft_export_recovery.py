from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

WORKER_PATH = Path("scripts/banking_v2/cloud_recover_continuation_export.py")
REMERGE_PATH = Path("scripts/banking_v2/hf_job_remerge_tool_sft.py")
JOB_PATH = Path("scripts/banking_v2/hf_job_recover_continuation_export.py")
LAUNCHER_PATH = Path(
    "scripts/banking_v2/run_remote_continuation_export_recovery.sh"
)


def _load_worker() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cloud_recover_continuation_export",
        WORKER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WORKER = _load_worker()


def test_recovery_uses_unchanged_parity_gates() -> None:
    assert WORKER.MINIMUM_ARGMAX_AGREEMENT == 0.999
    assert WORKER.MAXIMUM_LOGIT_DIFFERENCE == 0.3
    assert WORKER.MAXIMUM_P999_DIFFERENCE == 0.07


def test_recovery_candidate_order_starts_with_fp16_native() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    fp16_position = source.index('"merged_subdir": "merged-fp16-native"')
    bf16_position = source.index('"merged_subdir": "merged-fp32-bf16"')

    assert fp16_position < bf16_position
    assert '"merge_dtype": "float16"' in source[fp16_position:bf16_position]
    assert '"inference_dtype": "float16"' in source[fp16_position:bf16_position]


def test_recovery_publishes_only_after_validate_parity() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    run_body = source.split("def run_candidate", 1)[1].split("def publish", 1)[0]
    main_body = source.split("def main", 1)[1]

    assert "validate_parity(" in run_body
    assert main_body.index('if candidate["passed"]') < main_body.index(
        "publish(args, candidate=candidate"
    )
    assert main_body.index('candidate["release_dtype"] == "float16"') < (
        main_body.index("publish(args, candidate=candidate")
    )


def test_recovery_weights_and_evidence_use_one_atomic_hub_commit() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    publish_body = source.split("def publish", 1)[1].split("def main", 1)[0]

    assert "CommitOperationAdd(" in publish_body
    assert "api.create_commit(" in publish_body
    assert "api.upload_folder(" not in publish_body
    assert "api.upload_file(" not in publish_body
    assert '"training_metadata.json"' in publish_body
    assert '"merge_parity_diagnostics.json"' in publish_body
    assert '"training_result.json"' in publish_body
    assert "Returned by the atomic Hub commit" not in publish_body
    assert publish_body.index("weights_revision = str(release_commit.oid)") < (
        publish_body.index('commit_message="Record exact continuation release revision"')
    )


def test_remerge_dtype_defaults_preserve_existing_release_path() -> None:
    source = REMERGE_PATH.read_text(encoding="utf-8")

    assert 'default="float32"' in source
    assert 'default="float16"' in source
    assert 'default="fp16_remerge.json"' in source


def test_recovery_launcher_is_export_only_and_capped() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    job = JOB_PATH.read_text(encoding="utf-8")

    assert "--timeout 1h" in launcher
    assert "--volume hf://buckets/spkc83/jobs-artifacts:/data" in launcher
    assert "cloud_recover_continuation_export.py" in job
    assert "cloud_continue_tool_sft.py" not in job
    assert "trainer.train" not in job
    assert "rm " not in launcher


def test_recovery_rejects_symbolic_revisions() -> None:
    with pytest.raises(ValueError, match="exact 40-character"):
        WORKER.require_exact_revision("main", field="--recovery-source-commit")
