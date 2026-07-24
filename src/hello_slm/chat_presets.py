from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatPreset:
    id: str
    label: str
    prompt: str
    expected_answer: int
    operation: str
    supported: bool


ARITHMETIC_CHAT_PRESETS = (
    ChatPreset(
        id="add_small",
        label="Warm-up addition",
        prompt="What is 2 + 3?",
        expected_answer=5,
        operation="addition",
        supported=True,
    ),
    ChatPreset(
        id="add_carry",
        label="Addition with carry",
        prompt="What is 17 + 28?",
        expected_answer=45,
        operation="addition",
        supported=True,
    ),
    ChatPreset(
        id="subtract",
        label="Subtraction",
        prompt="What is 93 - 47?",
        expected_answer=46,
        operation="subtraction",
        supported=True,
    ),
    ChatPreset(
        id="negative",
        label="Negative result",
        prompt="What is 31 - 61?",
        expected_answer=-30,
        operation="subtraction",
        supported=True,
    ),
    ChatPreset(
        id="divide",
        label="Exact division",
        prompt="What is 144 / 12?",
        expected_answer=12,
        operation="division",
        supported=True,
    ),
    ChatPreset(
        id="multiply_limit",
        label="Limitation check",
        prompt="What is 12 * 8?",
        expected_answer=96,
        operation="multiplication",
        supported=False,
    ),
)
