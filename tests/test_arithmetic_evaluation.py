from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hello_slm.arithmetic_evaluation import (  # noqa: E402
    conversation_to_arithmetic_example,
    extract_final_integer,
    gate_arithmetic_scores,
    infer_operation,
    score_arithmetic_records,
    select_arithmetic_examples,
)
from hello_slm.cli import main  # noqa: E402
from hello_slm.data import Conversation, Message  # noqa: E402


def _conversation(
    conversation_id: str,
    question: str,
    answer: str,
    *,
    metadata: dict[str, object] | None = None,
) -> Conversation:
    return Conversation(
        schema_version=1,
        conversation_id=conversation_id,
        source="test",
        license="MIT",
        created_at="2026-07-22T00:00:00Z",
        messages=(
            Message(role="user", content=question, loss=False),
            Message(role="assistant", content=answer, loss=True),
        ),
        split="test",
        manifest_path="data/test.jsonl",
        metadata=metadata,
    )


def test_extract_final_integer_handles_signed_and_comma_numbers() -> None:
    assert extract_final_integer("reasoning gives -4, then final answer is -12.") == -12
    assert extract_final_integer("The total is 1,024 apples.") == 1024
    assert extract_final_integer("No numeric answer") is None


def test_conversation_example_prefers_metadata_and_falls_back_to_answer_text() -> None:
    explicit = _conversation(
        "a",
        "What is 2 + 3?",
        "5",
        metadata={"operation": "Addition", "expected_answer": 5},
    )
    fallback = _conversation("b", "What is 9 - 4?", "The answer is 5.")

    assert conversation_to_arithmetic_example(explicit) == {
        "conversation_id": "a",
        "operation": "addition",
        "prompt": "What is 2 + 3?",
        "expected_answer": 5,
    }
    assert conversation_to_arithmetic_example(fallback) == {
        "conversation_id": "b",
        "operation": "subtraction",
        "prompt": "What is 9 - 4?",
        "expected_answer": 5,
    }


def test_select_arithmetic_examples_is_sorted_and_stratified() -> None:
    conversations = [
        _conversation("mul.2", "What is 2 * 7?", "14"),
        _conversation("add.2", "What is 2 + 7?", "9"),
        _conversation("add.1", "What is 1 + 7?", "8"),
        _conversation("mul.1", "What is 1 * 7?", "7"),
    ]

    selected = select_arithmetic_examples(conversations, max_per_operation=1)

    assert [(item["operation"], item["conversation_id"]) for item in selected] == [
        ("addition", "add.1"),
        ("multiplication", "mul.1"),
    ]


def test_score_and_gate_require_overall_and_per_operation_accuracy() -> None:
    records = [
        {"operation": "addition", "correct": True, "parse_failed": False},
        {"operation": "addition", "correct": False, "parse_failed": False},
        {"operation": "division", "correct": False, "parse_failed": True},
    ]

    overall, by_operation = score_arithmetic_records(records)
    gate = gate_arithmetic_scores(overall, by_operation)

    assert overall == {"count": 3, "correct": 1, "accuracy": 1 / 3, "parse_failures": 1}
    assert by_operation["division"]["parse_failures"] == 1
    assert gate["passed"] is False
    assert "overall accuracy below 0.90" in gate["failures"]


def test_gate_fails_when_required_operation_is_missing() -> None:
    overall = {"count": 1, "correct": 1, "accuracy": 1.0, "parse_failures": 0}
    by_operation = {
        "add": {"count": 1, "correct": 1, "accuracy": 1.0, "parse_failures": 0}
    }

    gate = gate_arithmetic_scores(
        overall,
        by_operation,
        required_operations=("add", "subtract", "divide"),
    )

    assert gate["passed"] is False
    assert "required operation subtract has no selected examples" in gate["failures"]
    assert "required operation divide has no selected examples" in gate["failures"]


def test_infer_operation_unknown_is_explicit() -> None:
    assert infer_operation("How many marbles are there?") == "unknown"


def test_cli_eval_arithmetic_returns_nonzero_when_gate_fails(monkeypatch, tmp_path: Path) -> None:
    def fake_load_config(config_path: str, work_dir: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(artifact_dir=tmp_path)

    def fake_evaluate_arithmetic(*_: object, **__: object) -> dict[str, object]:
        return {"command": "eval-arithmetic", "status": "failed", "gate": {"passed": False}}

    monkeypatch.setattr("hello_slm.cli.load_config", fake_load_config)
    monkeypatch.setattr("hello_slm.cli.evaluate_arithmetic", fake_evaluate_arithmetic)

    exit_code = main(["eval-arithmetic", "--config", "unused.toml", "--checkpoint", "fake.pt"])

    assert exit_code == 3
