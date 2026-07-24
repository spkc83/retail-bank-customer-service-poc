from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hello_slm.banking_hf_generator import (
    BANKING_MODEL_ENV,
    HuggingFaceBankingGenerator,
    MissingBankingCheckpointError,
    _to_transformers_messages,
)
from hello_slm.banking_policy import ChatMessage, DomainRouteResult


def test_default_model_path_can_be_overridden_by_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "banking-model"
    monkeypatch.setenv(BANKING_MODEL_ENV, str(model_dir))

    generator = HuggingFaceBankingGenerator()

    assert generator.model_path == model_dir
    assert generator.loaded is False


def test_missing_checkpoint_fails_honestly_for_in_domain_generation(tmp_path: Path) -> None:
    generator = HuggingFaceBankingGenerator(tmp_path / "missing")

    with pytest.raises(
        MissingBankingCheckpointError,
        match="trained banking checkpoint is missing",
    ):
        generator.generate(
            [ChatMessage(role="user", content="How do I replace my card?")],
            seed=7,
            route=DomainRouteResult(
                route="in_domain",
                confidence=0.95,
                intent="card_support",
                reason="test",
            ),
        )


def test_rendered_transformers_messages_include_system_and_history() -> None:
    messages = _to_transformers_messages(
        [
            ChatMessage(role="user", content="I lost my card."),
            ChatMessage(role="assistant", content="I can help."),
            ChatMessage(role="user", content="What about the fee?"),
        ]
    )

    assert messages[0]["role"] == "system"
    assert messages[1:] == [
        {"role": "user", "content": "I lost my card."},
        {"role": "assistant", "content": "I can help."},
        {"role": "user", "content": "What about the fee?"},
    ]


def test_generator_uses_apply_chat_template_and_lazy_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calls: dict[str, int | bool] = {"tokenizer": 0, "model": 0, "template": 0, "generate": 0}

    class FakeTokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            add_generation_prompt: bool,
            return_tensors: str,
        ) -> torch.Tensor:
            calls["template"] = int(calls["template"]) + 1
            assert add_generation_prompt is True
            assert return_tensors == "pt"
            assert messages[-1] == {"role": "user", "content": "How do I open an account?"}
            return torch.tensor([[11, 12]], dtype=torch.long)

        def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            assert token_ids.tolist() == [42, 2]
            return "You can open an account online or at a branch."

    class FakeModel:
        def to(self, device: torch.device) -> FakeModel:
            return self

        def eval(self) -> None:
            calls["eval"] = True

        def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
            calls["generate"] = int(calls["generate"]) + 1
            assert input_ids.tolist() == [[11, 12]]
            assert kwargs["max_new_tokens"] == 160
            return torch.tensor([[11, 12, 42, 2]], dtype=torch.long)

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda path, trust_remote_code: (
                calls.update(tokenizer=int(calls["tokenizer"]) + 1) or FakeTokenizer()
            )
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda path, torch_dtype, trust_remote_code: (
                calls.update(model=int(calls["model"]) + 1) or FakeModel()
            )
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    generator = HuggingFaceBankingGenerator(model_dir, device="cpu")
    first = generator.generate(
        [ChatMessage(role="user", content="How do I open an account?")],
        seed=3101,
        route=DomainRouteResult(
            route="in_domain",
            confidence=0.95,
            intent="account_services",
            reason="test",
        ),
    )
    second = generator.generate(
        [ChatMessage(role="user", content="How do I open an account?")],
        seed=3102,
        route=DomainRouteResult(
            route="in_domain",
            confidence=0.95,
            intent="account_services",
            reason="test",
        ),
    )

    assert first == "You can open an account online or at a branch."
    assert second == "You can open an account online or at a branch."
    assert calls["tokenizer"] == 1
    assert calls["model"] == 1
    assert calls["template"] == 2
    assert calls["generate"] == 2
