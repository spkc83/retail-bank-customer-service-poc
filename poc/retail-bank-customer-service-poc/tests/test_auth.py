from __future__ import annotations

import pytest

from auth import load_demo_auth


def test_auth_secret_requires_exact_demo_usernames() -> None:
    credentials = load_demo_auth(
        '{"alex.demo":"alex-password","maya.demo":"maya-password"}'
    )

    assert credentials == [
        ("alex.demo", "alex-password"),
        ("maya.demo", "maya-password"),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "{}",
        '{"alex.demo":"same","maya.demo":"same"}',
        '{"alex.demo":"short","maya.demo":"long-enough-password"}',
        '{"alex.demo":"long-enough-password","other.demo":"long-enough-password"}',
    ],
)
def test_auth_secret_fails_closed_for_missing_or_weak_credentials(payload: str) -> None:
    with pytest.raises(ValueError):
        load_demo_auth(payload)
