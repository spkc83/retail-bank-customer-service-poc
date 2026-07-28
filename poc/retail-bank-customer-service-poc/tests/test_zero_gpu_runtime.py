from __future__ import annotations

import importlib
import sys

import pytest


def test_zero_gpu_boundary_is_stateless_and_fails_cleanly_without_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POC_SKIP_MODEL_LOAD", "1")
    sys.modules.pop("zero_gpu_runtime", None)
    runtime = importlib.import_module("zero_gpu_runtime")

    assert not hasattr(runtime, "BANK")
    assert not hasattr(runtime, "run_model_service")
    assert not hasattr(runtime.generate_final_answer, "_zero_gpu_config")
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime.generate_final_answer(
            [{"role": "user", "content": "Show my balance."}],
            {"list_accounts": {"accounts": []}},
            128,
        )
