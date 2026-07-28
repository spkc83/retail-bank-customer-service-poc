from __future__ import annotations

import json
import os
from typing import Any

DEMO_USERNAMES = ("alex.demo", "maya.demo")


def load_demo_auth(payload: str | None = None) -> list[tuple[str, str]]:
    raw = payload if payload is not None else os.environ.get("DEMO_AUTH_JSON", "")
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("DEMO_AUTH_JSON must be a JSON object") from error
    if not isinstance(parsed, dict) or set(parsed) != set(DEMO_USERNAMES):
        raise ValueError("DEMO_AUTH_JSON must define exactly the two demo usernames")
    passwords = []
    for username in DEMO_USERNAMES:
        password = parsed.get(username)
        if not isinstance(password, str) or len(password) < 12:
            raise ValueError("each demo password must contain at least 12 characters")
        passwords.append(password)
    if len(set(passwords)) != len(passwords):
        raise ValueError("demo accounts must use different passwords")
    return list(zip(DEMO_USERNAMES, passwords, strict=True))
