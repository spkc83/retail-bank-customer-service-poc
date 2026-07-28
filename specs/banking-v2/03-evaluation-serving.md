# Banking-v2 evaluation and serving

## Serving path

The public POC uses deterministic orchestration for backend selection. The
latest user turn and bounded conversation history pass through four gates:

```text
credential guard
  → CPU /chat dispatch and dual-head domain/intent router
  → deterministic capability planner
  → direct response, or registered ZeroGPU model event when needed
  → CPU synthetic backend plus stateless 9B finalizer
```

The router is an advisory classifier and audit signal, not a prompt suggestion
and not the component that selects backend operations. The deterministic
planner decides whether to answer directly, execute a supported read bundle,
execute one supported write, ask for clarification, return an unsupported
banking response, or return the exact out-of-domain response:

```text
I can only help with retail banking and financial-services questions. Please ask about accounts, cards, transfers, payments, loans, or related banking support.
```

The released router uses a shared DistilBERT encoder with a binary
supported-banking/OOD head and a 77-way Banking77 intent head. Its held-out
results are intent macro F1
`0.951208`, in-domain false-refusal rate `0.013689`, OOD false-accept rate
`0.007733`, and calibrated banking threshold `0.98`. These results are
diagnostic POC gates; deterministic capability evidence still controls the
backend path.

## Workflow rules

The planner implements the deployed capability contract:

- greetings and acknowledgements return direct conversational responses;
- explicit non-banking topics return the stock OOD response before ZeroGPU
  inference;
- unsupported banking requests return an honest POC-limitation response;
- read-only requests can bundle multiple supported reads in user-requested
  order;
- account-changing requests are limited to one explicit write in a turn;
- mixed read/write or multi-write requests return clarification and make no
  synthetic change;
- mailing-address history is limited to available synthetic service cases, not
  a full profile audit log.

The 9B model is a grounded response finalizer. It receives only sanitized,
verified workflow results and must not invent balances, dates, identifiers,
statuses, or actions. Server-side validation rejects empty, unsafe, or
internal-identifier-bearing final responses. For writes, finalizer failure
rolls back the synthetic backend transaction. Multi-read responses and
incomplete or contradictory factual drafts use a labeled deterministic
rendering of the verified workflow results.

## Multi-turn behavior

History is isolated per authenticated user and browser session. The finalizer
receives a bounded list of sanitized alternating user/assistant messages plus
the current user message. Assistant messages are stripped of UI diagnostics
before reuse. Tests must cover clarification, follow-up, user correction, and
an in-domain conversation that transitions to an out-of-domain request.

## Test-time scaling

Test-time scaling is not enabled in the deployed POC. Each backend-executing
request uses one deterministic 9B generation for final answer writing. A CPU
chat-dispatch event handles direct conversational, unsupported-banking, OOD,
credential-guard, and clarification responses without requesting a GPU. Only a
unique pending model turn triggers the separately registered ZeroGPU event.
The UI prevents another submit or reset while that event is pending, and a
session epoch causes stale queued turns to execute nothing.

Any future multi-candidate path must prove at least a two-point improvement on
the held-out composite score over the one-generation baseline. OOD false
accepts and in-domain false refusals may each regress by no more than 0.5
percentage points. The verifier may rank complete candidates but may not
synthesize a new answer. Seeds, decoding parameters, candidate text, scores,
selected index, router decision, workflow plan, and latency must be recorded
for reproducibility.

## Release gates

- exact canned response on 100% of accepted held-out OOD cases
- in-domain false-refusal rate at most 2%
- contextual banking-follow-up false-refusal rate at most 5%
- Banking77 intent-router macro F1 at least 0.90
- unresolved placeholder rate zero
- response/intent consistency at least 0.90
- multi-turn continuity score at least 0.85
- deterministic workflow accuracy at least 0.95 on supported POC intents
- zero committed writes on clarification, OOD, unsupported, unsafe, or
  unavailable-finalizer paths
- no PII-like strings in released training or evaluation files
- dense baseline and MoE evaluated on the same frozen splits

The public POC must display enough route, workflow, and backend-state evidence
to debug failures without exposing internal identifiers to the finalizer or to
the customer response. Presets are smoke tests only and must not be presented
as evidence of generalization.
