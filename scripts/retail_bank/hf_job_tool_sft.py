# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.12.0",
#   "bitsandbytes==0.50.0",
#   "datasets==4.5.0",
#   "huggingface-hub==1.22.0",
#   "peft==0.18.1",
#   "safetensors==0.8.0",
#   "torch>=2.9,<3",
#   "trackio>=0.33,<0.34",
#   "transformers==5.13.0",
#   "trl==0.26.2",
# ]
# ///
"""Bootstrap the pinned banking-v3 source inside a Hugging Face GPU Job."""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

SOURCE_REPO = "spkc83/retail-bank-servicing"
DATASET_REPO = "spkc83/retail-bank-agent-sft"
MODEL_REPO = "spkc83/retail-bank-agent-9b"
BASE_MODEL = "ibm-granite/granite-4.1-8b"
BASE_REVISION = "1504002f650e656a0a3789d99574df12e3e94ed0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir", default="/data/retail-bank-agent-9b")
    parser.add_argument("--resume-from")
    parser.add_argument("--max-steps", type=int, default=3_000)
    parser.add_argument("--max-train-seconds", type=int, default=14_400)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    return parser.parse_args()


def download_source(source_commit: str, destination: Path) -> Path:
    if not source_commit or any(char not in "0123456789abcdef" for char in source_commit):
        raise ValueError("--source-commit must be a lowercase hexadecimal Git commit")
    url = f"https://github.com/{SOURCE_REPO}/archive/{source_commit}.tar.gz"
    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "retail-bank-tool-sft-job"})
    with urllib.request.urlopen(request, timeout=120) as response:
        archive = response.read()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        bundle.extractall(destination, filter="data")
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError("source archive did not contain exactly one repository root")
    return roots[0]


def main() -> int:
    args = parse_args()
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN must be passed as a Hugging Face Job secret")
    with tempfile.TemporaryDirectory(prefix="retail-bank-agent-source-") as temp_dir:
        temp_root = Path(temp_dir)
        source_root = download_source(args.source_commit, temp_root / "source")
        dataset_root = Path(
            snapshot_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                revision=args.dataset_revision,
                local_dir=temp_root / "dataset",
                token=os.environ["HF_TOKEN"],
            )
        )
        manifest = Path(args.manifest) if args.manifest else dataset_root / "manifest.json"
        if not manifest.is_file():
            raise RuntimeError(f"dataset manifest is unavailable: {manifest}")
        env = {
            **os.environ,
            "PYTHONPATH": str(source_root / "src"),
            "RETAIL_BANK_ALLOW_REMOTE_TOOL_SFT": "banking-v3-tool-sft",
            "RETAIL_BANK_SOURCE_COMMIT": args.source_commit,
            "RETAIL_BANK_TOOL_SFT_DATASET_REPO": DATASET_REPO,
            "RETAIL_BANK_TOOL_SFT_DATASET_REVISION": args.dataset_revision,
        }
        command = [
            sys.executable,
            str(source_root / "scripts/retail_bank/cloud_train_tool_sft.py"),
            "--execute-remote",
            "--allow-remote-execution",
            "--push-to-hub",
            "--manifest",
            str(manifest),
            "--output-dir",
            args.output_dir,
            "--hub-dest",
            MODEL_REPO,
            "--base-model",
            BASE_MODEL,
            "--base-revision",
            BASE_REVISION,
            "--family",
            "granite",
            "--precision",
            "bf16-lora",
            "--max-steps",
            str(args.max_steps),
            "--max-train-seconds",
            str(args.max_train_seconds),
            "--batch-size",
            "2",
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--max-seq-len",
            "2048",
            "--learning-rate",
            "1e-4",
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--trackio-project",
            "retail-bank-agent-v3",
            "--trackio-run-name",
            f"granite-tool-sft-{args.source_commit[:8]}",
        ]
        if args.resume_from:
            command.extend(["--resume-from", args.resume_from])
        subprocess.run(command, cwd=source_root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
