# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.12.0",
#   "huggingface-hub==1.22.0",
#   "peft==0.18.1",
#   "safetensors==0.8.0",
#   "torch>=2.9,<3",
#   "transformers==5.13.0",
# ]
# ///
"""Bootstrap an export-only continuation recovery on Hugging Face Jobs."""

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

SOURCE_REPO = "spkc83/retail-bank-servicing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-source-commit", required=True)
    parser.add_argument("--training-source-commit", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--parent-model-revision", required=True)
    parser.add_argument("--training-job", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selected-adapter-subdir", required=True)
    parser.add_argument("--selected-step", type=int, required=True)
    return parser.parse_args()


def require_exact_revision(value: str, *, field: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be an exact 40-character lowercase Git revision")


def download_source(source_commit: str, destination: Path) -> Path:
    require_exact_revision(source_commit, field="--recovery-source-commit")
    url = f"https://github.com/{SOURCE_REPO}/archive/{source_commit}.tar.gz"
    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "retail-bank-continuation-export-recovery"},
    )
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
    require_exact_revision(
        args.training_source_commit,
        field="--training-source-commit",
    )
    require_exact_revision(args.dataset_revision, field="--dataset-revision")
    require_exact_revision(args.parent_model_revision, field="--parent-model-revision")
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN must be passed as a Hugging Face Job secret")
    with tempfile.TemporaryDirectory(prefix="retail-bank-export-recovery-") as temp_dir:
        source_root = download_source(
            args.recovery_source_commit,
            Path(temp_dir) / "source",
        )
        command = [
            sys.executable,
            str(
                source_root
                / "scripts/retail_bank/cloud_recover_continuation_export.py"
            ),
            "--output-root",
            args.output_root,
            "--recovery-source-commit",
            args.recovery_source_commit,
            "--training-source-commit",
            args.training_source_commit,
            "--dataset-revision",
            args.dataset_revision,
            "--parent-model-revision",
            args.parent_model_revision,
            "--training-job",
            args.training_job,
            "--selected-adapter-subdir",
            args.selected_adapter_subdir,
            "--selected-step",
            str(args.selected_step),
        ]
        subprocess.run(command, cwd=source_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
