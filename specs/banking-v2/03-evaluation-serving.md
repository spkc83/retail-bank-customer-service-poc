# Banking-v2 evaluation and serving

## Serving experiment

The public POC evaluates a model-driven dual-head-router plus 9B-agent design:

```text
direct registered ZeroGPU chat event
  → credential-value guard
  → CPU-resident dual-head classifier inside the managed worker
  → OOD stock response, or
  → 9B model with intent guidance and token-budgeted history
  → direct response or one generated batch of Qwen tool calls
  → synthetic backend tool results
  → second 9B generation for the final response
```

The domain head produces `in_domain`, `uncertain`, or `out_of_domain`.
`p(in_domain) >= 0.98` is confidently in-domain, `p(in_domain) < 0.50` is OOD
by the binary head's decision boundary, and the middle region is uncertain.
OOD bypasses the 9B generator, although every turn enters the managed ZeroGPU
event. The intent head always exposes its top three predictions; they are
included as advisory model context rather than mapped to backend workflows.

## Model and tool ownership

For allowed and uncertain turns, the 9B model owns natural conversation,
contextual interpretation, clarification, tool selection, user-facing tool
arguments, and final response generation.

The runtime performs mechanical Qwen `<tool_call>` parsing and direct invocation
of generated mock functions. It does not contain a regex capability planner,
semantic authorization validator, deterministic grounded repair, or
CPU-generated servicing fallback.

One first-pass generation may contain up to eight ordered calls. Malformed tool
protocol fails the model turn. Unknown tool names, unsupported arguments, and
backend errors are returned as structured tool results for the second
generation. A plain first-pass response completes without tool execution.

## Context contract

The complete valid transcript is retained per authenticated browser session.
Canonical state contains user, assistant, assistant-tool-call, tool-result, and
final-assistant messages.

The operating input budget is 8,192 tokenizer-measured tokens, with 512 new
tokens reserved for generation. Context selection retains the system prompt and
current router guidance, the complete latest interaction, and newest earlier
complete interactions while they fit. A tool-call chain is never split. An
oversized latest interaction fails rather than being silently truncated.

The model checkpoint's hard context limit is 32,768 tokens; the lower operating
budget controls ZeroGPU latency and memory.

## Observability and evaluation

The UI reports domain probabilities, top-three intent predictions, generated
tool calls and arguments, tool status, response path, and current synthetic
backend state.

UI presets and prior screenshots are regression examples only. The held-out
evaluation must cover paraphrases, conversational follow-ups, intent-head
errors, OOD boundaries, multi-tool selection, write arguments, grounding, and
naturalness.

## Failure behavior

ZeroGPU allocation, generation, token-budget, or protocol failure returns an
honest model-unavailable answer. It never substitutes a deterministic banking
response. Mock tools execute when generated. Since this is a behavioral
experiment without a rollback policy, a successful synthetic write remains
visible if second-pass generation later fails.

## Release gates

- Banking77 intent macro F1 at least 0.90;
- in-domain false-refusal rate at most 2%;
- held-out OOD false-accept rate at most 1%;
- greetings and customer-service small talk reach the 9B model;
- tool-name accuracy at least 0.95 on supported held-out scenarios;
- tool-argument accuracy at least 0.90;
- multi-turn reference-resolution accuracy at least 0.85;
- malformed tool-call rate below 1%;
- model-authored response rate 100% for successful allowed turns;
- no CPU-generated servicing answer during model unavailability;
- complete canonical tool chains are retained or omitted as whole units;
- dense baseline and MoE are evaluated on the same frozen scenarios.

An authenticated live ZeroGPU tool round trip is required to claim hosted model
inference. A running Space that only reaches the failure handler does not pass
that gate.
