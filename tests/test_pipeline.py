from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hello_slm.cli import main  # noqa: E402
from hello_slm.config import canonical_sha256, load_experiment_config  # noqa: E402
from hello_slm.evaluation import evaluate  # noqa: E402
from hello_slm.generation import generate_reply  # noqa: E402
from hello_slm.training import (  # noqa: E402
    PipelineError,
    _resolve_precision_runtime,
    build_dataset,
    build_tokenizer,
    load_config,
    train,
    validate_run,
)


def test_full_smoke_flow_runs_in_tmp_path(tmp_path: Path) -> None:
    work_dir = tmp_path / "artifacts"
    exit_code = main(
        [
            "smoke",
            "--config",
            str(ROOT / "configs" / "smoke.toml"),
            "--work-dir",
            str(work_dir),
        ]
    )

    assert exit_code == 0
    latest = work_dir / "checkpoints" / "latest.pt"
    assert latest.exists()
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 3
    assert checkpoint["training_state"]["dataloader_cursor"] > 0
    assert checkpoint["fingerprints"]["effective_config_hash"] == load_config(
        ROOT / "configs" / "smoke.toml", work_dir
    ).effective_hash
    assert (work_dir / "tokenizer" / "tokenizer.json").exists()
    assert (work_dir / "dataset" / "manifest.json").exists()
    assert (work_dir / "reports" / "eval.json").exists()


def test_pipeline_primitives_train_resume_eval_and_generate(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "smoke.toml", tmp_path / "artifacts")

    validate_report = validate_run(config)
    tokenizer_report = build_tokenizer(config)
    dataset_report = build_dataset(config)
    first = train(config, max_steps=2)
    resumed = train(config, max_steps=3, resume=first["checkpoint"])
    eval_report = evaluate(config, checkpoint_path=resumed["checkpoint"])
    chat = generate_reply(config, checkpoint_path=resumed["checkpoint"], prompt="Hello")

    assert validate_report["status"] == "success"
    assert tokenizer_report["tokenizer_fingerprint"]
    assert dataset_report["dataset_fingerprint"]
    assert first["global_step"] == 2
    assert resumed["global_step"] == 3
    assert first["precision_runtime"] == {
        "requested_device": "cpu",
        "requested_precision": "float32",
        "device": "cpu",
        "precision": "float32",
        "autocast_enabled": False,
        "grad_scaler_enabled": False,
        "fallback_applied": False,
        "fallback_reason": None,
    }
    assert eval_report["release_eligible"] is False
    assert eval_report["heldout_loss"] > 0
    assert eval_report["heldout_perplexity"] > 0
    assert 0.0 <= eval_report["assistant_token_accuracy"] <= 1.0
    assert chat["metadata"]["generated_tokens"] <= config.data["generation"]["max_new_tokens"]


def test_strict_resume_rejects_effective_config_mismatch(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "smoke.toml", tmp_path / "artifacts")
    validate_run(config)
    build_tokenizer(config)
    build_dataset(config)
    first = train(config, max_steps=2)

    changed_data = copy.deepcopy(config.data)
    changed_data["training"]["optimizer"]["learning_rate"] = 0.002
    changed = replace(config, data=changed_data, effective_hash=canonical_sha256(changed_data))

    with pytest.raises(PipelineError, match="effective_config_hash mismatch"):
        train(changed, max_steps=3, resume=first["checkpoint"])


def test_chat_rejects_characters_outside_restricted_vocabulary(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "smoke.toml", tmp_path / "artifacts")
    validate_run(config)
    build_tokenizer(config)
    build_dataset(config)
    trained = train(config, max_steps=1)

    with pytest.raises(PipelineError, match="disallowed characters"):
        generate_reply(config, checkpoint_path=trained["checkpoint"], prompt="Hello @")


def test_work_dir_redirect_does_not_mutate_source_config() -> None:
    base = load_experiment_config(ROOT / "configs" / "smoke.toml", ROOT)
    redirected = load_config(ROOT / "configs" / "smoke.toml", ROOT / "tmp-artifacts")

    assert base.data["run"]["artifact_dir"] == "artifacts/smoke"
    assert redirected.data["run"]["artifact_dir"] == str((ROOT / "tmp-artifacts").resolve())
    assert base.effective_hash != redirected.effective_hash


def test_precision_runtime_keeps_cpu_float32_without_config_mutation() -> None:
    config = load_experiment_config(ROOT / "configs" / "smoke.toml", ROOT)
    before = copy.deepcopy(config.data["training"])

    runtime = _resolve_precision_runtime(config)

    assert runtime.to_mapping() == {
        "requested_device": "cpu",
        "requested_precision": "float32",
        "device": "cpu",
        "precision": "float32",
        "autocast_enabled": False,
        "grad_scaler_enabled": False,
        "fallback_applied": False,
        "fallback_reason": None,
    }
    assert config.data["training"] == before


def test_precision_runtime_rejects_required_cpu_mixed_precision() -> None:
    config = load_experiment_config(ROOT / "configs" / "smoke.toml", ROOT)
    data = copy.deepcopy(config.data)
    data["training"]["precision"] = "float16"
    changed = replace(config, data=data, effective_hash=canonical_sha256(data))

    with pytest.raises(PipelineError, match="CPU training requires float32 precision"):
        _resolve_precision_runtime(changed)


def test_precision_runtime_allows_cpu_precision_fallback_without_mutation() -> None:
    config = load_experiment_config(ROOT / "configs" / "smoke.toml", ROOT)
    data = copy.deepcopy(config.data)
    data["training"]["precision"] = "float16"
    data["training"]["allow_precision_fallback"] = True
    changed = replace(config, data=data, effective_hash=canonical_sha256(data))

    runtime = _resolve_precision_runtime(changed)

    assert runtime.device == "cpu"
    assert runtime.precision == "float32"
    assert runtime.fallback_applied is True
    assert changed.data["training"]["precision"] == "float16"


def test_structural_validation_accepts_focused_profile_without_training(tmp_path: Path) -> None:
    exit_code = main(
        [
            "validate",
            "--config",
            str(ROOT / "configs" / "focused-125m.toml"),
            "--work-dir",
            str(tmp_path / "focused"),
            "--structural",
        ]
    )

    assert exit_code == 0


def test_chat_cli_accepts_bounded_generation_override(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "smoke.toml", tmp_path / "artifacts")
    validate_run(config)
    build_tokenizer(config)
    build_dataset(config)
    trained = train(config, max_steps=1)

    exit_code = main(
        [
            "chat",
            "--config",
            str(ROOT / "configs" / "smoke.toml"),
            "--work-dir",
            str(config.artifact_dir),
            "--checkpoint",
            trained["checkpoint"],
            "--prompt",
            "Hello",
            "--max-new-tokens",
            "4",
            "--json",
        ]
    )

    assert exit_code == 0


def test_checkpoint_digest_is_verified_before_load(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "smoke.toml", tmp_path / "artifacts")
    validate_run(config)
    build_tokenizer(config)
    build_dataset(config)
    trained = train(config, max_steps=1)
    checkpoint = Path(trained["checkpoint"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")

    with pytest.raises(PipelineError, match="byte-count mismatch"):
        evaluate(config, checkpoint_path=checkpoint)
