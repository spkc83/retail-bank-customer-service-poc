from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hello_slm.banking_chat_runtime import (
    BankingChatRuntime,
    CandidateScore,
    TTSEvalMetrics,
    count_complete_turns,
    select_bounded_context,
)
from hello_slm.banking_chat_runtime import (
    TestTimeScalingPolicy as ScalingPolicy,
)
from hello_slm.banking_policy import (
    OOD_STOCK_RESPONSE,
    ChatMessage,
    DeterministicBankingRouter,
    DomainRouteResult,
)


@dataclass
class RecordingGenerator:
    calls: list[int]

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        seed: int,
        route: DomainRouteResult,
    ) -> str:
        self.calls.append(seed)
        return f"candidate-{seed}"


class SeedPreferenceVerifier:
    def score(
        self,
        candidate: str,
        messages: Sequence[ChatMessage],
        *,
        route: DomainRouteResult,
    ) -> CandidateScore:
        score = 2.0 if candidate.endswith("3103") else 1.0
        return CandidateScore(
            total=score,
            intent_domain_consistency=1.0,
            policy=1.0,
            context_continuity=1.0,
            repetition=1.0,
        )


def test_ood_returns_exact_stock_response_without_generator_call() -> None:
    generator = RecordingGenerator(calls=[])
    runtime = BankingChatRuntime(generator=generator)

    reply = runtime.reply("s1", "What is the weather tomorrow?")

    assert reply.response == OOD_STOCK_RESPONSE
    assert reply.route == "out_of_domain"
    assert reply.selected_index is None
    assert reply.candidates == ()
    assert generator.calls == []


def test_follow_up_uses_full_session_history_for_domain_routing() -> None:
    generator = RecordingGenerator(calls=[])
    runtime = BankingChatRuntime(generator=generator)

    first = runtime.reply("s1", "I lost my debit card. What should I do?")
    second = runtime.reply("s1", "What about the fee?")

    assert first.route == "in_domain"
    assert second.route == "in_domain"
    assert second.confidence_band == "medium"
    assert len(generator.calls) == 2


def test_context_truncation_preserves_system_and_complete_turns_only() -> None:
    messages = [
        ChatMessage(role="system", content="banking assistant"),
        ChatMessage(role="user", content="turn 1 user account"),
        ChatMessage(role="assistant", content="turn 1 assistant"),
        ChatMessage(role="user", content="orphan old user"),
        ChatMessage(role="user", content="turn 2 user card"),
        ChatMessage(role="assistant", content="turn 2 assistant"),
        ChatMessage(role="assistant", content="orphan assistant"),
        ChatMessage(role="user", content="current question"),
    ]

    selected = select_bounded_context(messages, max_complete_turns=1)

    assert selected == (
        ChatMessage(role="system", content="banking assistant"),
        ChatMessage(role="user", content="turn 2 user card"),
        ChatMessage(role="assistant", content="turn 2 assistant"),
        ChatMessage(role="user", content="current question"),
    )
    assert count_complete_turns(selected) == 1


def test_session_isolation_supports_ood_then_in_domain_and_in_then_ood() -> None:
    generator = RecordingGenerator(calls=[])
    runtime = BankingChatRuntime(generator=generator)

    assert runtime.reply("s1", "How do I dispute a debit card charge?").route == "in_domain"
    assert runtime.reply("s2", "What about the fee?").route == "out_of_domain"
    assert runtime.reply("s2", "Can I open a savings account?").route == "in_domain"
    assert runtime.reply("s1", "Write Python code for me.").route == "out_of_domain"


def test_high_confidence_in_domain_uses_one_candidate() -> None:
    generator = RecordingGenerator(calls=[])
    runtime = BankingChatRuntime(
        generator=generator,
        tts_policy=ScalingPolicy(
            enabled=True,
            eval_metrics=TTSEvalMetrics(
                composite_improvement_pp=5.0,
                ood_false_accept_regression_pp=0.0,
                in_domain_false_refusal_regression_pp=0.0,
            ),
        ),
    )

    reply = runtime.reply("s1", "How do I transfer money between accounts?")

    assert reply.confidence_band == "high"
    assert reply.candidates == ("candidate-3101",)
    assert generator.calls == [3101]


def test_medium_confidence_tts_uses_four_deterministic_candidates_and_verifier_selection() -> None:
    generator = RecordingGenerator(calls=[])
    runtime = BankingChatRuntime(
        generator=generator,
        verifier=SeedPreferenceVerifier(),
        tts_policy=ScalingPolicy(
            enabled=True,
            eval_metrics=TTSEvalMetrics(
                composite_improvement_pp=2.0,
                ood_false_accept_regression_pp=0.5,
                in_domain_false_refusal_regression_pp=0.5,
            ),
        ),
    )

    runtime.reply("s1", "I lost my debit card.")
    reply = runtime.reply("s1", "What about the fee?")

    assert reply.tts_enabled is True
    assert reply.confidence_band == "medium"
    assert reply.candidates == (
        "candidate-3101",
        "candidate-3102",
        "candidate-3103",
        "candidate-3104",
    )
    assert reply.selected_index == 2
    assert reply.response == "candidate-3103"


def test_tts_disabled_by_regression_gate_forces_one_candidate() -> None:
    generator = RecordingGenerator(calls=[])
    runtime = BankingChatRuntime(
        generator=generator,
        tts_policy=ScalingPolicy(
            enabled=True,
            eval_metrics=TTSEvalMetrics(
                composite_improvement_pp=4.0,
                ood_false_accept_regression_pp=0.6,
                in_domain_false_refusal_regression_pp=0.0,
            ),
        ),
    )

    runtime.reply("s1", "I lost my debit card.")
    reply = runtime.reply("s1", "What about the fee?")

    assert reply.tts_enabled is False
    assert reply.confidence_band == "medium"
    assert reply.candidates == ("candidate-3101",)


def test_default_router_is_explicitly_non_production_baseline() -> None:
    assert "non-production" in (DeterministicBankingRouter.__doc__ or "")
