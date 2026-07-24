from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hello_slm import chat_runtime
from hello_slm.arithmetic_evaluation import extract_final_integer
from hello_slm.chat_presets import ARITHMETIC_CHAT_PRESETS
from hello_slm.chat_runtime import ArithmeticChatRuntime

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "arithmetic-curriculum-30m.toml"
CHECKPOINT = ROOT / "artifacts" / "arithmetic-curriculum-30m" / "checkpoints" / "latest.pt"


def test_arithmetic_chat_presets_are_unique_and_label_supported_scope() -> None:
    assert len({preset.id for preset in ARITHMETIC_CHAT_PRESETS}) == len(
        ARITHMETIC_CHAT_PRESETS
    )
    assert len({preset.prompt for preset in ARITHMETIC_CHAT_PRESETS}) == len(
        ARITHMETIC_CHAT_PRESETS
    )
    supported = {preset.operation for preset in ARITHMETIC_CHAT_PRESETS if preset.supported}
    unsupported = {preset.operation for preset in ARITHMETIC_CHAT_PRESETS if not preset.supported}

    assert supported == {"addition", "subtraction", "division"}
    assert unsupported == {"multiplication"}


def test_runtime_is_lazy_and_reuses_loaded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = ArithmeticChatRuntime(CONFIG, CHECKPOINT)
    calls = {"load": 0, "generate": 0}

    def fake_load(*args: object, **kwargs: object) -> tuple[object, dict[str, int], dict]:
        calls["load"] += 1
        return SimpleNamespace(config=SimpleNamespace(max_seq_len=128)), {"global_step": 17}, {}

    def fake_generate(*args: object, **kwargs: object) -> str:
        calls["generate"] += 1
        return "2 + 3 = 5."

    monkeypatch.setattr(chat_runtime, "load_model_from_checkpoint", fake_load)
    monkeypatch.setattr(chat_runtime, "load_tokenizer", lambda path: object())
    monkeypatch.setattr(chat_runtime, "generate_arithmetic_response", fake_generate)
    monkeypatch.setattr(chat_runtime, "sha256_file", lambda path: "digest")
    monkeypatch.setattr(Path, "exists", lambda self: True)

    assert runtime.loaded is False
    first = runtime.reply("What is 2 + 3?")
    second = runtime.reply("What is 17 + 28?")

    assert runtime.loaded is True
    assert first.response == "2 + 3 = 5."
    assert second.global_step == 17
    assert calls == {"load": 1, "generate": 2}


def test_shiny_ui_contains_chat_presets_and_limitations_without_loading_model() -> None:
    from hello_slm import shiny_app

    html = str(shiny_app.app_ui)

    assert shiny_app.MODEL_RUNTIME.loaded is False
    assert 'id="model_chat"' in html
    for preset in ARITHMETIC_CHAT_PRESETS:
        assert f'id="preset_{preset.id}"' in html
    assert "Model limitations" in html
    assert "Multiplication is exploratory" in html


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="trained arithmetic checkpoint is not present")
def test_real_trained_checkpoint_answers_small_addition() -> None:
    runtime = ArithmeticChatRuntime(CONFIG, CHECKPOINT)

    reply = runtime.reply("What is 2 + 3?")

    assert extract_final_integer(reply.response) == 5
    assert reply.global_step == 2000
