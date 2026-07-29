from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_training_module() -> ModuleType:
    path = Path("scripts/banking_v2/train_dual_head_router.py")
    spec = importlib.util.spec_from_file_location("banking_router_training", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_prioritizes_ood_specificity_under_recall_constraint() -> None:
    training = load_training_module()
    probabilities = [0.99, 0.90, 0.80, 0.60, 0.55, 0.40, 0.20, 0.10]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]

    calibration = training.calibrate_threshold(
        probabilities,
        labels,
        minimum_in_domain_recall=0.75,
    )

    assert calibration["threshold"] > 0.55
    assert calibration["in_domain_recall"] >= 0.75
    assert calibration["ood_specificity"] == 1.0


def test_calibration_enforces_supported_conversation_recall() -> None:
    training = load_training_module()
    probabilities = [0.99, 0.90, 0.80, 0.50, 0.55, 0.40, 0.20, 0.10]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    example_kinds = [
        "banking77_single",
        "banking77_single",
        "banking77_single",
        "clinc_conversational_in_domain",
        "clinc_nonbanking",
        "clinc_nonbanking",
        "clinc_nonbanking",
        "clinc_nonbanking",
    ]

    calibration = training.calibrate_threshold(
        probabilities,
        labels,
        example_kinds=example_kinds,
        minimum_in_domain_recall=0.75,
        minimum_conversational_recall=1.0,
    )

    assert calibration["threshold"] <= 0.50
    assert calibration["conversational_recall"] == 1.0


def test_metrics_and_release_gates_cover_transition_subsets() -> None:
    training = load_training_module()
    metrics = training.evaluate_predictions(
        domain_probabilities=[0.95, 0.90, 0.85, 0.05, 0.10],
        domain_labels=[1, 1, 1, 0, 0],
        intent_predictions=[0, 1, 0, 0, 1],
        intent_labels=[0, 1, -100, -100, -100],
        example_kinds=[
            "banking77_single",
            "same_intent_followup",
            "clinc_conversational_in_domain",
            "clinc_nonbanking",
            "banking_to_ood_transition",
        ],
        threshold=0.5,
        num_intents=2,
    )

    assert metrics["intent_macro_f1"] == 1.0
    assert metrics["in_domain_false_refusal_rate"] == 0.0
    assert metrics["ood_false_accept_rate"] == 0.0
    assert metrics["followup_false_refusal_rate"] == 0.0
    assert metrics["conversational_false_refusal_rate"] == 0.0
    assert metrics["transition_ood_false_accept_rate"] == 0.0
    assert training.release_gate_failures(metrics) == []


def test_release_gate_reports_each_failed_contract() -> None:
    training = load_training_module()
    failures = training.release_gate_failures(
        {
            "intent_macro_f1": 0.80,
            "in_domain_false_refusal_rate": 0.04,
            "ood_false_accept_rate": 0.08,
            "followup_false_refusal_rate": 0.07,
            "conversational_false_refusal_rate": 0.08,
            "transition_ood_false_accept_rate": 0.06,
        }
    )

    assert len(failures) == 6
