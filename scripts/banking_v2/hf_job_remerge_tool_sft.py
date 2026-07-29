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
"""Rebuild a trained LoRA merge with explicit accumulation and release dtypes."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "ibm-granite/granite-4.1-8b"
BASE_REVISION = "1504002f650e656a0a3789d99574df12e3e94ed0"
DEFAULT_OUTPUT_ROOT = Path("/mnt/artifacts/retail-bank-agent-9b-3a6a7efe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-subdir", default="merged-fp16")
    parser.add_argument("--adapter-subdir", default="adapter")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument(
        "--merge-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--release-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--report-name", default="fp16_remerge.json")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN must be passed as a Hugging Face Job secret")
    adapter_dir = args.output_root / args.adapter_subdir
    merged_dir = args.output_root / args.output_subdir
    if merged_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing merge directory: {merged_dir}")
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"trained adapter is unavailable: {adapter_dir}")

    started = time.monotonic()
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    merge_dtype = dtypes[args.merge_dtype]
    release_dtype = dtypes[args.release_dtype]
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        dtype=merge_dtype,
        device_map={"": torch.cuda.current_device()},
    )
    adapter_model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        autocast_adapter_dtype=True,
    )
    adapter_model.eval()
    merged = adapter_model.merge_and_unload(safe_merge=True)
    merged.to(dtype=release_dtype)
    merged.config.torch_dtype = release_dtype
    merged.save_pretrained(
        merged_dir,
        safe_serialization=True,
        max_shard_size="20GB",
    )
    tokenizer.save_pretrained(merged_dir)
    report = {
        "contract": "banking-v3-fp16-remerge/v1",
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "merge_accumulation_dtype": args.merge_dtype,
        "release_weight_dtype": args.release_dtype,
        "safe_merge": True,
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(args.output_root / args.report_name, report)
    print(json.dumps(report, sort_keys=True))
    del adapter_model
    del merged
    del base
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
