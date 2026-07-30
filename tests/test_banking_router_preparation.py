from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_preparation_module() -> ModuleType:
    path = Path("scripts/retail_bank/prepare_dual_head_router_data.py")
    spec = importlib.util.spec_from_file_location("banking_router_preparation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_banking77_csv_parser_preserves_release_row_identity() -> None:
    preparation = load_preparation_module()
    payloads = {
        "train": b"text,category\r\nWhere is my card?,card_arrival\r\n",
        "test": b"text,category\r\nWhy is it late?,card_arrival\r\n",
    }

    rows = preparation.parse_banking77_csvs(payloads)

    assert rows == [
        {
            "source_row_id": 0,
            "split": "train",
            "text": "Where is my card?",
            "label": "card_arrival",
            "source_dataset": "PolyAI/banking77",
            "source_revision": preparation.BANKING77_RELEASE_REVISION,
            "license": "CC-BY-4.0",
            "trainable": False,
        },
        {
            "source_row_id": 0,
            "split": "test",
            "text": "Why is it late?",
            "label": "card_arrival",
            "source_dataset": "PolyAI/banking77",
            "source_revision": preparation.BANKING77_RELEASE_REVISION,
            "license": "CC-BY-4.0",
            "trainable": False,
        },
    ]


def test_release_split_digest_check_reports_drift() -> None:
    preparation = load_preparation_module()
    expected = {
        "prepared_split_sha256": {
            "train": "a" * 64,
            "validation": "b" * 64,
            "test": "c" * 64,
        }
    }
    actual = [
        {"name": "train", "sha256": "a" * 64},
        {"name": "validation", "sha256": "b" * 64},
        {"name": "test", "sha256": "d" * 64},
    ]

    with pytest.raises(ValueError, match="test split digest drift"):
        preparation.verify_release_split_digests(actual, expected)
