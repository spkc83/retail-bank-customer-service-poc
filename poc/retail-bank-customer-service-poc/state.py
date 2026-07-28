from __future__ import annotations

import os
from pathlib import Path

from mock_bank import SessionBankRegistry

ROOT = Path(__file__).resolve().parent
BANK = SessionBankRegistry.from_json(
    ROOT / "synthetic_bank.json",
    ttl_seconds=7200,
    max_sessions=32,
    database_dir=Path(
        os.environ.get("POC_SESSION_DB_DIR", "/tmp/retail-bank-servicing-poc")
    ),
)
