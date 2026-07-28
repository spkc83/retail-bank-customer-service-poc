from __future__ import annotations

import importlib
from typing import Any

import pytest


def router_result(
    *,
    route: str = "out_of_domain",
    intent: str | None = None,
    banking_probability: float = 0.05,
    intent_confidence: float = 0.05,
) -> dict[str, Any]:
    return {
        "route": route,
        "banking_probability": banking_probability,
        "confidence": banking_probability if route == "in_domain" else 1 - banking_probability,
        "intent": intent,
        "intent_confidence": intent_confidence,
        "threshold": 0.98,
        "router_revision": "test-router",
    }


@pytest.fixture
def orchestration_module():
    try:
        return importlib.import_module("orchestration")
    except ModuleNotFoundError as error:
        if error.name != "orchestration":
            raise
        pytest.fail(
            "Expected poc/retail-bank-customer-service-poc/orchestration.py "
            "with plan_workflow(message, history, router_result)."
        )


def plan_for(
    orchestration_module,
    message: str,
    *,
    history: list[dict[str, Any]] | None = None,
    router: dict[str, Any] | None = None,
):
    return orchestration_module.plan_workflow(
        message,
        history or [],
        router or router_result(),
    )


def assert_no_tools(plan) -> None:
    assert plan.read_tools == ()
    assert plan.write_tool is None
    assert plan.arguments == {}


@pytest.mark.parametrize(
    ("message", "history"),
    [
        ("Hello, how are you?", []),
        (
            "Thanks",
            [
                {"role": "user", "content": "Show my balances."},
                {"role": "assistant", "content": "Your checking account is available."},
            ],
        ),
    ],
)
def test_planner_returns_conversational_response_without_tools(
    orchestration_module,
    message: str,
    history: list[dict[str, Any]],
) -> None:
    plan = plan_for(
        orchestration_module,
        message,
        history=history,
        router=router_result(
            route="in_domain",
            intent="cash_withdrawal",
            banking_probability=0.94,
            intent_confidence=0.80,
        ),
    )

    assert plan.category == "conversational"
    assert_no_tools(plan)
    assert isinstance(plan.direct_response, str)
    assert plan.direct_response.strip()


def test_planner_returns_out_of_domain_for_weather_even_when_router_accepts_it(
    orchestration_module,
) -> None:
    plan = plan_for(
        orchestration_module,
        "What is the weather in Dallas today?",
        router=router_result(
            route="in_domain",
            intent="cash_withdrawal",
            banking_probability=0.99,
            intent_confidence=0.88,
        ),
    )

    assert plan.category == "out_of_domain"
    assert_no_tools(plan)
    assert isinstance(plan.direct_response, str)
    assert "retail-banking" in plan.direct_response


def test_planner_returns_unsupported_banking_for_mortgage_opening(
    orchestration_module,
) -> None:
    plan = plan_for(
        orchestration_module,
        "Can you help me open a mortgage account?",
        router=router_result(
            route="in_domain",
            intent="cash_withdrawal",
            banking_probability=0.99,
            intent_confidence=0.80,
        ),
    )

    assert plan.category == "unsupported_banking"
    assert_no_tools(plan)
    assert isinstance(plan.direct_response, str)
    direct_response = plan.direct_response.lower()
    assert "cannot" in direct_response or "not supported" in direct_response


@pytest.mark.parametrize(
    ("message", "expected_tool"),
    [
        ("Show my account balances.", "list_accounts"),
        ("What recent transactions are on my account?", "list_transactions"),
        ("What is my debit card status?", "list_cards"),
        ("Show my pending transfers.", "list_transfers"),
        ("When was my mailing address changed?", "list_service_cases"),
    ],
)
def test_planner_rescues_supported_single_read_when_router_is_wrong_or_low_confidence(
    orchestration_module,
    message: str,
    expected_tool: str,
) -> None:
    plan = plan_for(
        orchestration_module,
        message,
        router=router_result(
            route="out_of_domain",
            intent=None,
            banking_probability=0.32,
            intent_confidence=0.10,
        ),
    )

    assert plan.category == "single_read"
    assert plan.read_tools == (expected_tool,)
    assert plan.write_tool is None
    assert plan.arguments == {}
    assert plan.direct_response is None


@pytest.mark.parametrize(
    ("message", "expected_tools"),
    [
        (
            "Show transfers and recent transactions.",
            ("list_transfers", "list_transactions"),
        ),
        (
            "Show my balances and card status.",
            ("list_accounts", "list_cards"),
        ),
    ],
)
def test_planner_orders_and_deduplicates_compound_read_tools(
    orchestration_module,
    message: str,
    expected_tools: tuple[str, ...],
) -> None:
    plan = plan_for(
        orchestration_module,
        message,
        router=router_result(
            route="in_domain",
            intent="pending_transfer",
            banking_probability=0.74,
            intent_confidence=0.35,
        ),
    )

    assert plan.category == "multi_read"
    assert plan.read_tools == expected_tools
    assert plan.write_tool is None
    assert plan.arguments == {}
    assert plan.direct_response is None


@pytest.mark.parametrize(
    ("message", "expected_tool", "expected_arguments"),
    [
        ("Cancel pending transfer to River Consulting.", "cancel_transfer", {}),
        ("Cancel the completed transfer to Jamie Lee.", "cancel_transfer", {}),
        ("Freeze my debit card ending in 4821.", "freeze_card", {"last4": "4821"}),
        ("Replace my debit card ending in 4821.", "replace_card", {"last4": "4821"}),
        ("Dispute the Lumina Market transaction.", "dispute_transaction", {}),
    ],
)
def test_planner_rescues_one_explicit_write_when_router_is_wrong_or_low_confidence(
    orchestration_module,
    message: str,
    expected_tool: str,
    expected_arguments: dict[str, Any],
) -> None:
    plan = plan_for(
        orchestration_module,
        message,
        router=router_result(
            route="out_of_domain",
            intent=None,
            banking_probability=0.21,
            intent_confidence=0.07,
        ),
    )

    assert plan.category == "single_write"
    assert plan.read_tools == ()
    assert plan.write_tool == expected_tool
    assert plan.arguments == expected_arguments
    assert plan.direct_response is None


@pytest.mark.parametrize(
    "message",
    [
        "Show my transfers and cancel the River Consulting transfer.",
        "Freeze my card and show recent transactions.",
    ],
)
def test_planner_clarifies_mixed_read_and_write_without_executing_any_tool(
    orchestration_module,
    message: str,
) -> None:
    plan = plan_for(
        orchestration_module,
        message,
        router=router_result(
            route="in_domain",
            intent="pending_transfer",
            banking_probability=0.99,
        ),
    )

    assert plan.category == "clarification"
    assert_no_tools(plan)
    assert isinstance(plan.direct_response, str)
    assert plan.direct_response.strip()


def test_planner_clarifies_multiple_writes_without_executing_any_tool(
    orchestration_module,
) -> None:
    plan = plan_for(
        orchestration_module,
        "Freeze my debit card and cancel the River Consulting transfer.",
        router=router_result(route="in_domain", intent="cancel_transfer", banking_probability=0.99),
    )

    assert plan.category == "clarification"
    assert_no_tools(plan)
    assert isinstance(plan.direct_response, str)
    assert plan.direct_response.strip()


@pytest.mark.parametrize(
    ("message", "expected_category", "expected_reads", "expected_write"),
    [
        (
            "cancel pending transfer to River Consulting",
            "single_write",
            (),
            "cancel_transfer",
        ),
        (
            "show transfers and recent transactions",
            "multi_read",
            ("list_transfers", "list_transactions"),
            None,
        ),
        (
            "when was my mailing address changed",
            "single_read",
            ("list_service_cases",),
            None,
        ),
    ],
)
def test_planner_handles_screenshot_regression_prompts_without_router_dependence(
    orchestration_module,
    message: str,
    expected_category: str,
    expected_reads: tuple[str, ...],
    expected_write: str | None,
) -> None:
    plan = plan_for(
        orchestration_module,
        message,
        router=router_result(
            route="out_of_domain",
            intent=None,
            banking_probability=0.73,
            intent_confidence=0.20,
        ),
    )

    assert plan.category == expected_category
    assert plan.read_tools == expected_reads
    assert plan.write_tool == expected_write
    assert plan.arguments == {}


def test_planner_does_not_rescue_nonsupported_text_from_router_intent_alone(
    orchestration_module,
) -> None:
    plan = plan_for(
        orchestration_module,
        "Tell me about consulting firms near a river.",
        router=router_result(
            route="in_domain",
            intent="cancel_transfer",
            banking_probability=0.99,
            intent_confidence=0.99,
        ),
    )

    assert plan.category == "out_of_domain"
    assert_no_tools(plan)
    assert isinstance(plan.direct_response, str)
    assert "retail-banking" in plan.direct_response


def test_planner_preserves_router_evidence_on_supported_rescue(
    orchestration_module,
) -> None:
    router = router_result(
        route="out_of_domain",
        intent=None,
        banking_probability=0.737093,
        intent_confidence=0.05,
    )
    plan = plan_for(
        orchestration_module,
        "Cancel pending transfer to River Consulting.",
        router=router,
    )

    assert plan.category == "single_write"
    assert plan.write_tool == "cancel_transfer"
    assert plan.router_route == "out_of_domain"
    assert plan.router_intent is None
    assert "explicit" in plan.reason.lower()


def test_planner_uses_recent_assistant_context_for_write_follow_up(
    orchestration_module,
) -> None:
    plan = plan_for(
        orchestration_module,
        "Freeze it.",
        history=[
            {"role": "user", "content": "What is my debit card status?"},
            {
                "role": "assistant",
                "content": (
                    "Your synthetic debit card is active.\n\n"
                    "---\n_Model workflow: `list_cards` · revision `abc…`_"
                ),
            },
        ],
        router=router_result(
            route="out_of_domain",
            banking_probability=0.4,
        ),
    )

    assert plan.category == "single_write"
    assert plan.write_tool == "freeze_card"
    assert plan.read_tools == ()
