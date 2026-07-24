"""Small, explicit helpers for reproducible and atomic local artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_path(target: Path) -> tuple[int, Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    return descriptor, Path(temporary)


def atomic_write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    descriptor, temporary = _atomic_path(target)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(value))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_torch_save(path: str | Path, value: Any) -> Path:
    target = Path(path)
    descriptor, temporary = _atomic_path(target)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
        keys_array = np.asarray(keys, dtype=np.uint32).copy()
        state["numpy"] = {
            "name": name,
            "keys": torch.from_numpy(keys_array),
            "position": position,
            "has_gauss": has_gauss,
            "cached_gaussian": cached_gaussian,
        }
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])
    if "numpy" in state:
        try:
            import numpy as np

            numpy_state = state["numpy"]
            np.random.set_state(
                (
                    numpy_state["name"],
                    numpy_state["keys"].cpu().numpy(),
                    numpy_state["position"],
                    numpy_state["has_gauss"],
                    numpy_state["cached_gaussian"],
                )
            )
        except ImportError as exc:
            message = "checkpoint requires NumPy RNG state but NumPy is unavailable"
            raise RuntimeError(message) from exc


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
