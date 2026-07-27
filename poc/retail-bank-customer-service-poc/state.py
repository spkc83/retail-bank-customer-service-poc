from __future__ import annotations

from pathlib import Path

from mock_bank import SessionBankRegistry

ROOT = Path(__file__).resolve().parent
BANK = SessionBankRegistry.from_json(
    ROOT / "synthetic_bank.json",
    ttl_seconds=7200,
    max_sessions=32,
)
