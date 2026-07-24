from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hello_slm.arithmetic_evaluation import generate_arithmetic_response
from hello_slm.artifacts import sha256_file
from hello_slm.config import ExperimentConfig
from hello_slm.tokenizer import load_tokenizer
from hello_slm.training import PipelineError, load_config, load_model_from_checkpoint


@dataclass(frozen=True)
class ChatReply:
    response: str
    latency_seconds: float
    device: str
    global_step: int
    checkpoint_sha256: str
    tokenizer_sha256: str


class ArithmeticChatRuntime:
    """Lazy, process-local runtime for the trained arithmetic chat checkpoint."""

    def __init__(self, config_path: str | Path, checkpoint_path: str | Path) -> None:
        self.config: ExperimentConfig = load_config(config_path)
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: torch.device | None = None
        self._global_step: int | None = None
        self._checkpoint_sha256: str | None = None
        self._tokenizer_sha256: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def parameter_count(self) -> int:
        return self.config.parameter_count

    @property
    def profile(self) -> str:
        return str(self.config.data["run"]["id"])

    def reply(self, prompt: str) -> ChatReply:
        started = time.perf_counter()
        with self._lock:
            self._ensure_loaded()
            if self._model is None or self._tokenizer is None or self._device is None:
                raise PipelineError("arithmetic chat runtime did not initialize")
            response = generate_arithmetic_response(
                self.config,
                model=self._model,
                tokenizer=self._tokenizer,
                device=self._device,
                prompt=prompt,
            )
            if not response:
                response = "The model returned an empty response."
            return ChatReply(
                response=response,
                latency_seconds=time.perf_counter() - started,
                device=str(self._device),
                global_step=int(self._global_step or 0),
                checkpoint_sha256=str(self._checkpoint_sha256),
                tokenizer_sha256=str(self._tokenizer_sha256),
            )

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        if not self.checkpoint_path.exists():
            raise PipelineError(f"trained checkpoint is missing: {self.checkpoint_path}")
        tokenizer_path = self.config.artifact_dir / "tokenizer" / "tokenizer.json"
        if not tokenizer_path.exists():
            raise PipelineError(f"trained tokenizer is missing: {tokenizer_path}")
        requested = str(self.config.data["training"]["device"])
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        model, checkpoint, _ = load_model_from_checkpoint(
            self.config,
            self.checkpoint_path,
            device=device,
        )
        self._model = model
        self._tokenizer = load_tokenizer(tokenizer_path)
        self._device = device
        self._global_step = int(checkpoint["global_step"])
        self._checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self._tokenizer_sha256 = sha256_file(tokenizer_path)
