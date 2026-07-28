from __future__ import annotations

from pathlib import Path

from hello_slm.banking_hf_generator import HuggingFaceBankingGenerator
from hello_slm.banking_policy import OOD_STOCK_RESPONSE


def test_banking_shiny_ui_contains_presets_and_status_without_loading_model() -> None:
    from hello_slm import banking_shiny_app

    html = str(banking_shiny_app.app_ui)

    generator = banking_shiny_app.MODEL_RUNTIME.generator
    assert isinstance(generator, HuggingFaceBankingGenerator)
    assert generator.loaded is False
    assert 'id="banking_chat"' in html
    for preset in banking_shiny_app.BANKING_CHAT_PRESETS:
        assert f'id="banking_preset_{preset.id}"' in html
        assert preset.prompt in html
    assert "Route" in html
    assert "Confidence" in html
    assert "Candidates" in html
    assert OOD_STOCK_RESPONSE in html


def test_banking_app_default_model_path_is_banking_v2_artifact() -> None:
    from hello_slm import banking_shiny_app

    expected = (
        Path(__file__).resolve().parents[1] / "artifacts" / "banking-v2-moe-9b" / "final"
    )
    assert expected == banking_shiny_app.MODEL_PATH
