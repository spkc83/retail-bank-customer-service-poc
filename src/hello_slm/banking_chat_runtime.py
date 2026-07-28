from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from hello_slm.banking_policy import (
    OOD_STOCK_RESPONSE,
    BankingDomainClassifier,
    ChatMessage,
    DeterministicBankingRouter,
    DomainRouteResult,
)

ConfidenceBand = Literal["high", "medium", "low"]


class BankingCandidateGenerator(Protocol):
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        seed: int,
        route: DomainRouteResult,
    ) -> str:
        ...


@dataclass(frozen=True)
class CandidateScore:
    total: float
    intent_domain_consistency: float
    policy: float
    context_continuity: float
    repetition: float


class BankingCandidateVerifier(Protocol):
    def score(
        self,
        candidate: str,
        messages: Sequence[ChatMessage],
        *,
        route: DomainRouteResult,
    ) -> CandidateScore:
        ...


@dataclass(frozen=True)
class TTSEvalMetrics:
    composite_improvement_pp: float
    ood_false_accept_regression_pp: float
    in_domain_false_refusal_regression_pp: float


@dataclass(frozen=True)
class TestTimeScalingPolicy:
    enabled: bool = False
    eval_metrics: TTSEvalMetrics | None = None
    high_confidence_threshold: float = 0.80
    medium_confidence_threshold: float = 0.50
    medium_candidates: int = 4

    @property
    def active(self) -> bool:
        metrics = self.eval_metrics
        return (
            self.enabled
            and metrics is not None
            and metrics.composite_improvement_pp >= 2.0
            and metrics.ood_false_accept_regression_pp <= 0.5
            and metrics.in_domain_false_refusal_regression_pp <= 0.5
        )

    def band(self, confidence: float) -> ConfidenceBand:
        if confidence >= self.high_confidence_threshold:
            return "high"
        if confidence >= self.medium_confidence_threshold:
            return "medium"
        return "low"

    def candidate_count(self, route: DomainRouteResult) -> int:
        if route.route == "out_of_domain" or self.band(route.confidence) == "low":
            return 0
        if self.band(route.confidence) == "medium" and self.active:
            return self.medium_candidates
        return 1


@dataclass(frozen=True)
class BankingChatReply:
    response: str
    route: str
    domain_confidence: float
    candidates: tuple[str, ...]
    selected_index: int | None
    history_turns: int
    tts_enabled: bool
    confidence_band: ConfidenceBand
    intent: str | None


class DeterministicBankingCandidateGenerator:
    """Non-production placeholder for wiring tests before a trained banking SLM exists."""

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        seed: int,
        route: DomainRouteResult,
    ) -> str:
        latest = next(
            (message.content for message in reversed(messages) if message.role == "user"), ""
        )
        return f"[seed={seed}] Banking support response for {route.intent or 'banking'}: {latest}"


class DeterministicBankingVerifier:
    """Non-production deterministic verifier for test-time scaling integration tests."""

    def score(
        self,
        candidate: str,
        messages: Sequence[ChatMessage],
        *,
        route: DomainRouteResult,
    ) -> CandidateScore:
        words = candidate.lower().split()
        repeated = len(words) - len(set(words))
        repetition = max(0.0, 1.0 - (repeated / max(len(words), 1)))
        policy = 1.0 if route.route == "in_domain" and OOD_STOCK_RESPONSE not in candidate else 0.0
        continuity = 1.0 if any(message.role == "user" for message in messages) else 0.0
        consistency = 1.0 if "bank" in candidate.lower() or "support" in candidate.lower() else 0.5
        total = (consistency * 0.35) + (policy * 0.30) + (continuity * 0.20) + (
            repetition * 0.15
        )
        return CandidateScore(
            total=total,
            intent_domain_consistency=consistency,
            policy=policy,
            context_continuity=continuity,
            repetition=repetition,
        )


@dataclass
class BankingChatRuntime:
    classifier: BankingDomainClassifier = field(default_factory=DeterministicBankingRouter)
    generator: BankingCandidateGenerator = field(
        default_factory=DeterministicBankingCandidateGenerator
    )
    verifier: BankingCandidateVerifier = field(default_factory=DeterministicBankingVerifier)
    tts_policy: TestTimeScalingPolicy = field(default_factory=TestTimeScalingPolicy)
    max_complete_turns: int = 4
    seed_base: int = 3101
    _sessions: dict[str, list[ChatMessage]] = field(default_factory=dict)

    def reply(self, session_id: str, user_message: str) -> BankingChatReply:
        if not user_message.strip():
            raise ValueError("user_message must not be empty")
        history = self._sessions.setdefault(session_id, [])
        history.append(ChatMessage(role="user", content=user_message))
        context = select_bounded_context(history, max_complete_turns=self.max_complete_turns)
        route = self.classifier.classify(context)
        band = self.tts_policy.band(route.confidence)
        count = self.tts_policy.candidate_count(route)

        if count == 0:
            response = OOD_STOCK_RESPONSE
            history.append(ChatMessage(role="assistant", content=response))
            return BankingChatReply(
                response=response,
                route=route.route,
                domain_confidence=route.confidence,
                candidates=(),
                selected_index=None,
                history_turns=count_complete_turns(context),
                tts_enabled=self.tts_policy.active,
                confidence_band=band,
                intent=route.intent,
            )

        candidates = tuple(
            self.generator.generate(context, seed=self.seed_base + index, route=route)
            for index in range(count)
        )
        scores = tuple(
            self.verifier.score(candidate, context, route=route) for candidate in candidates
        )
        selected_index = max(range(len(scores)), key=lambda index: (scores[index].total, -index))
        response = candidates[selected_index]
        history.append(ChatMessage(role="assistant", content=response))
        return BankingChatReply(
            response=response,
            route=route.route,
            domain_confidence=route.confidence,
            candidates=candidates,
            selected_index=selected_index,
            history_turns=count_complete_turns(context),
            tts_enabled=self.tts_policy.active,
            confidence_band=band,
            intent=route.intent,
        )


def select_bounded_context(
    messages: Sequence[ChatMessage],
    *,
    max_complete_turns: int,
) -> tuple[ChatMessage, ...]:
    if max_complete_turns < 0:
        raise ValueError("max_complete_turns must be non-negative")
    system_messages = tuple(message for message in messages if message.role == "system")
    trailing_user = _trailing_user(messages)
    complete_turns = _complete_turns(messages)
    selected_turns = complete_turns[-max_complete_turns:] if max_complete_turns else []
    selected: list[ChatMessage] = list(system_messages)
    for user, assistant in selected_turns:
        selected.extend((user, assistant))
    if trailing_user is not None:
        selected.append(trailing_user)
    return tuple(selected)


def count_complete_turns(messages: Sequence[ChatMessage]) -> int:
    return len(_complete_turns(messages))


def _complete_turns(messages: Sequence[ChatMessage]) -> list[tuple[ChatMessage, ChatMessage]]:
    turns: list[tuple[ChatMessage, ChatMessage]] = []
    pending_user: ChatMessage | None = None
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            pending_user = message
            continue
        if message.role == "assistant" and pending_user is not None:
            turns.append((pending_user, message))
            pending_user = None
    return turns


def _trailing_user(messages: Sequence[ChatMessage]) -> ChatMessage | None:
    if messages and messages[-1].role == "user":
        return messages[-1]
    return None
