# /// script
# dependencies = [
#   "accelerate>=1.12,<2",
#   "huggingface-hub>=1.4,<2",
#   "jsonschema>=4.18",
#   "torch>=2.9,<3",
#   "transformers>=5.13,<5.14",
# ]
# ///
"""Bootstrap the banking-v2 MoE worker inside a Hugging Face Job."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

HUB_DEST = "spkc83/retail-bank-servicing-moe-9b"
ROOT = Path("/tmp/retail-bank-model-development")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_archive", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--per-layer-aux-coef", type=float, required=True)
    return parser.parse_args()


def publish_status(
    api: HfApi,
    *,
    run_id: str,
    source_commit: str,
    per_layer_aux_coef: float,
    stage: str,
    **extra: Any,
) -> None:
    payload = {
        "run_id": run_id,
        "stage": stage,
        "source_commit": source_commit,
        "hardware": "rtx-pro-6000",
        "world_size": 1,
        "timeout_seconds": 18_000,
        "max_steps": 1_000,
        "training_seed": 7_101,
        "per_layer_router_aux_loss_coef": per_layer_aux_coef,
        **extra,
    }
    api.create_repo(repo_id=HUB_DEST, repo_type="model", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=io.BytesIO(json.dumps(payload, indent=2, sort_keys=True).encode()),
        path_in_repo=f"runs/{run_id}/status.json",
        repo_id=HUB_DEST,
        repo_type="model",
        commit_message=f"{run_id}: {stage}",
    )


def run_checked(command: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> int:
    args = parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    status_args = {
        "api": api,
        "run_id": args.run_id,
        "source_commit": args.source_commit,
        "per_layer_aux_coef": args.per_layer_aux_coef,
    }

    try:
        publish_status(**status_args, stage="bootstrapping")
        ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.source_archive, mode="r:gz") as bundle:
            bundle.extractall(ROOT, filter="data")

        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        publish_status(**status_args, stage="preparing_data")
        run_checked([sys.executable, "-m", "hello_slm.banking_data", "audit-sources"], env=env)
        run_checked([sys.executable, "-m", "hello_slm.banking_data", "prepare"], env=env)

        publish_status(**status_args, stage="training")
        run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/banking_v2/cloud_train_banking_moe.py"),
                "--execute-remote",
                "--allow-remote-execution",
                "--push-to-hub",
                "--hub-dest",
                HUB_DEST,
                "--manifest",
                str(ROOT / "data/banking-v2/manifest.json"),
                "--output-dir",
                str(ROOT / "artifacts/banking-v2-moe-9b"),
                "--max-steps",
                "1000",
                "--batch-size",
                "1",
                "--max-seq-len",
                "512",
                "--learning-rate",
                "2e-5",
                "--checkpoint-every",
                "250",
            ],
            env=env,
        )
        publish_status(**status_args, stage="completed")
    except BaseException as exc:
        try:
            publish_status(
                **status_args,
                stage="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc()[-12_000:],
            )
        finally:
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
