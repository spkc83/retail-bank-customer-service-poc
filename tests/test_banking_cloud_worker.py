from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

WORKER_PATH = Path("scripts/banking_v2/cloud_train_banking_moe.py")


def _load_worker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cloud_train_banking_moe", WORKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load cloud worker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


worker = _load_worker()
BANKING_V2_OOD_STOCK_RESPONSE = worker.BANKING_V2_OOD_STOCK_RESPONSE
SimpleBankingTokenizer = worker.SimpleBankingTokenizer
WorkerConfig = worker.WorkerConfig
assert_remote_execution_allowed = worker.assert_remote_execution_allowed
build_dry_run_plan = worker.build_dry_run_plan
load_manifest_records = worker.load_manifest_records
remote_execution_allowed = worker.remote_execution_allowed
tokenize_chat_records = worker.tokenize_chat_records
ExpertHealthWindow = worker.ExpertHealthWindow
expert_health_failure_message = worker.expert_health_failure_message


def _worker_config(
    tmp_path: Path, *, allow_remote: bool = False, execute_remote: bool = False
) -> Any:
    return WorkerConfig(
        manifest=tmp_path / "manifest.json",
        output_dir=tmp_path / "out",
        max_steps=1,
        batch_size=1,
        max_seq_len=48,
        learning_rate=1e-4,
        checkpoint_every=1,
        resume_from=None,
        dry_run=not execute_remote,
        run_tiny_smoke=False,
        allow_remote_execution=allow_remote,
        push_to_hub=False,
        hub_dest="spkc83/hello-banking-moe-9b",
    )


def test_remote_execution_requires_flag_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _worker_config(tmp_path, allow_remote=True, execute_remote=True)
    monkeypatch.delenv("HELLO_SLM_ALLOW_REMOTE_TRAINING", raising=False)

    assert not remote_execution_allowed(config)
    with pytest.raises(PermissionError, match="HELLO_SLM_ALLOW_REMOTE_TRAINING=banking-v2"):
        assert_remote_execution_allowed(config)

    monkeypatch.setenv("HELLO_SLM_ALLOW_REMOTE_TRAINING", "banking-v2")
    assert remote_execution_allowed(config)


def test_dry_run_plan_declares_no_unguarded_remote_actions(tmp_path: Path) -> None:
    plan = build_dry_run_plan(_worker_config(tmp_path))

    assert plan["worker"] == "cloud_train_banking_moe"
    assert plan["remote_guard"]["requires_flag"] == "--allow-remote-execution"
    assert plan["remote_guard"]["requires_execution_switch"] == "--execute-remote"
    assert plan["pins"]["base_revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert "download 9B or dense base weights" in plan["will_not_do_without_guard"]
    assert plan["training_summary"]["generative_sft_dataset"] == "data/banking-v2/manifest.json"


def test_manifest_loader_and_chat_tokenization(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    record = {
        "messages": [
            {"role": "user", "content": "How do I replace a card?"},
            {"role": "assistant", "content": "I can help with card replacement."},
        ]
    }
    train_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    validation_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = {
        "splits": {
            "train": {"path": "train.jsonl"},
            "validation": {"path": "validation.jsonl"},
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    records = load_manifest_records(manifest_path, "train")
    examples = tokenize_chat_records(records, SimpleBankingTokenizer(), max_seq_len=96)

    assert records == [record]
    assert examples[0]["input_ids"].shape[0] == 96
    assert examples[0]["attention_mask"].sum().item() > 0
    assert examples[0]["labels"].shape == examples[0]["input_ids"].shape
    assert (examples[0]["labels"] == -100).any()
    assert (examples[0]["labels"] != -100).any()


def test_manifest_loader_supports_banking_v2_contract(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    record = {
        "messages": [
            {"role": "user", "content": "How do I replace a card?", "loss": False},
            {"role": "assistant", "content": "Use card services.", "loss": True},
        ]
    }
    train_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "contract": "banking-v2-manifest",
                "generative_sft": [
                    {
                        "name": "train",
                        "path": "data/banking-v2/train.jsonl",
                        "included_for_generative_sft": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert load_manifest_records(manifest_path, "train") == [record]


def test_expert_health_window_accumulates_assignments_across_steps() -> None:
    window = ExpertHealthWindow.empty(num_experts=28, target_steps=28)
    for expert_id in range(28):
        window.add(
            {0: worker.torch.full((16, 2), expert_id, dtype=worker.torch.long)},
            aux_loss=worker.torch.tensor(0.01),
            routed_down_grad_nonzero={0: expert_id == 4},
        )

    health = window.health()

    assert window.steps == 28
    assert window.ready is True
    assert health[0].min_assignment_fraction == pytest.approx(1 / 28)
    assert health[0].max_assignment_fraction == pytest.approx(1 / 28)
    assert health[0].routed_down_grad_nonzero is True
    assert worker.expert_health_passed(health)


def test_expert_health_failure_message_includes_layer_health_json() -> None:
    window = ExpertHealthWindow.empty(num_experts=28, target_steps=1)
    window.add(
        {0: worker.torch.zeros((16, 2), dtype=worker.torch.long)},
        aux_loss=worker.torch.tensor(float("nan")),
        routed_down_grad_nonzero={0: False},
    )
    health = [worker.asdict(item) for item in window.health()]

    message = expert_health_failure_message(
        step=250,
        health_window=window,
        last_health=health,
    )

    assert "expert-health gate failed at optimizer step 250" in message
    assert "window_steps=1" in message
    assert "layer_health=" in message
    assert '"aux_loss_finite": false' in message
    assert '"max_assignment_fraction": 1.0' in message


def test_expert_health_windows_evaluate_and_reset_independently() -> None:
    balanced = worker.torch.arange(28, dtype=worker.torch.long).reshape(14, 2)
    first = ExpertHealthWindow.empty(num_experts=28, target_steps=2)
    for _ in range(2):
        first.add(
            {0: balanced},
            aux_loss=worker.torch.tensor(0.01),
            routed_down_grad_nonzero={0: True},
        )

    assert first.ready
    assert worker.expert_health_passed(first.health())
    with pytest.raises(RuntimeError, match="must be evaluated"):
        first.add(
            {0: balanced},
            aux_loss=worker.torch.tensor(0.01),
            routed_down_grad_nonzero={0: True},
        )

    second = ExpertHealthWindow.empty(num_experts=28, target_steps=2)
    for _ in range(2):
        second.add(
            {0: worker.torch.zeros((14, 2), dtype=worker.torch.long)},
            aux_loss=worker.torch.tensor(0.01),
            routed_down_grad_nonzero={0: True},
        )

    assert second.ready
    assert not worker.expert_health_passed(second.health())


def test_tiny_smoke_cli_trains_and_writes_checkpoint(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/banking_v2/cloud_train_banking_moe.py",
            "--run-tiny-smoke",
            "--output-dir",
            str(tmp_path / "worker"),
            "--max-steps",
            "1",
            "--checkpoint-every",
            "1",
            "--max-seq-len",
            "32",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    metadata_path = tmp_path / "worker" / "checkpoints" / "step-000001" / "metadata.json"

    assert payload["steps"] == 1
    assert payload["trainable_counts"]["trainable"] > 0
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["ood_stock_response"] == BANKING_V2_OOD_STOCK_RESPONSE
    assert metadata["base_revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def test_worker_cli_default_is_dry_run() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/banking_v2/cloud_train_banking_moe.py"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["mode"] == "dry_run"
    assert payload["remote_guard"]["currently_allowed"] is False
    assert "write to Hugging Face Hub" in payload["will_not_do_without_guard"]
