# Banking-v2 evaluation and serving

## Domain router

The serving boundary routes the latest complete user turn together with bounded
conversation history. A low-confidence or out-of-domain decision bypasses model
generation and returns the exact canned response:

```text
I can only help with retail banking and financial-services questions. Please ask about accounts, cards, transfers, payments, loans, or related banking support.
```

The router is a safety boundary, not a prompt suggestion. The current local
keyword router is only a deterministic integration fixture; production
acceptance requires a trained and calibrated router evaluated on Banking77 plus
held-out non-financial prompts.

## Multi-turn behavior

History is isolated per session and consists only of complete user/assistant
turns. Truncation removes the oldest complete turn pair and never leaves an
orphan assistant message. Tests must cover clarification, follow-up, user
correction, and an in-domain conversation that transitions to an out-of-domain
request.

## Test-time scaling

Candidate scaling is conditional:

- low router confidence or OOD: zero generations; return the canned response
- high confidence: one generation
- medium confidence: four deterministic candidates followed by a verifier

Enable the four-candidate path only when it improves the held-out composite
score by at least two percentage points over one candidate. OOD false accepts
and in-domain false refusals may each regress by no more than 0.5 percentage
points. Otherwise deploy one candidate.

The verifier may rank complete candidates but may not synthesize a new answer.
Seeds, decoding parameters, candidate text, scores, selected index, router
decision, and latency are recorded for reproducibility.

## Release gates

- exact canned response on 100% of accepted held-out OOD cases
- in-domain false-refusal rate at most 2%
- contextual banking-follow-up false-refusal rate at most 5%
- Banking77 intent-router macro F1 at least 0.90
- unresolved placeholder rate zero
- response/intent consistency at least 0.90
- multi-turn continuity score at least 0.85
- no PII-like strings in released training or evaluation files
- dense baseline and MoE evaluated on the same frozen splits

The Shiny test application must display route, confidence, candidate count, and
turn history metadata. Presets are smoke tests only and must not be presented as
evidence of generalization.
