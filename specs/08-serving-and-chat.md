# Serving and chat specification

This document defines the bounded local chat behavior for release bundles. The
reference implementation is a local demonstration interface, not a network
service.

## Chat template

All training, evaluation, and serving MUST render conversations with the same
template:

```text
<|bos|><|system|>
{system_message}<|end|>
<|user|>
{user_message}<|end|>
<|assistant|>
{assistant_message}<|end|><|eos|>
```

If no system message exists, rendering starts with `<|bos|><|user|>`. For
multi-turn conversations, repeat user and assistant blocks in order. Generation
starts after the final `<|assistant|>` marker and stops at `<|end|>`, `<|eos|>`,
or the configured token budget. Serving MUST NOT emit `<|eos|>` before the first
assistant `<|end|>` for the generated turn.

Required special tokens:

| Token | Purpose |
|---|---|
| `<|pad|>` | Padding and ignored loss positions. |
| `<|unk|>` | Unknown token handling. |
| `<|bos|>` | Start of rendered conversation. |
| `<|eos|>` | End of rendered conversation. |
| `<|system|>` | System message boundary. |
| `<|user|>` | User message boundary. |
| `<|assistant|>` | Assistant message boundary. |
| `<|end|>` | End of one message block. |

Special token ids MUST be fixed by the tokenizer artifact and MUST NOT change
between training and serving.

## Default system message

Release bundles MUST include this default system message unless the config
overrides it:

```text
You are Hello SLM, a small local chat model trained from a restricted corpus. Answer only within the corpus and task instructions. If the answer is unsupported, say you do not know.
```

The system message is a behavioral hint, not a security boundary. Safety gates
MUST NOT rely on the system message alone.

## Bounded generation

`hello-slm chat --config PATH --checkpoint PATH` MUST enforce:

| Setting | Smoke default | Target default | Hard limit |
|---|---:|---:|---:|
| `max_context_tokens` | 128 | 1024 | 2048 |
| `max_new_tokens` | 32 | 160 | 256 |
| `temperature` | 0.7 | 0.7 | `0.0 <= value <= 1.5` |
| `top_k` | 20 | 40 | `1 <= value <= vocab_size` |
| `top_p` | 1.0 | 0.95 | `0.0 < value <= 1.0` |
| `repetition_penalty` | 1.0 | 1.05 | `1.0 <= value <= 2.0` |

The command MUST reject requests whose rendered prompt exceeds
`max_context_tokens`. It MUST stop on `<|end|>` or `<|eos|>`, trim special
tokens from user visible output, and report when output ended because of the
token budget.

## Decoding modes

Evaluation uses deterministic greedy decoding. Interactive chat MAY use sampling
within the bounded settings above. The release manifest MUST record the serving
defaults used for any sample transcripts.

## Input handling

The chat command MUST:

- accept UTF-8 input;
- normalize line endings to `\n`;
- reject empty user messages;
- reject messages exceeding configured character or token limits;
- treat user text as plain text, not as commands;
- avoid network, file-system corpus lookup, shell execution, or tool calls.

The restricted corpus is already encoded into the model weights. Serving MUST
NOT dynamically retrieve undeclared documents unless a future retrieval spec is
added.

## Output handling

The command MUST print only the assistant response and machine-readable metadata
when `--json` is requested. Metadata MUST include token counts, stop reason,
latency, decoding settings, checkpoint digest, and tokenizer digest.

Stop reasons are:

- `end_token`;
- `eos_token`;
- `max_new_tokens`;
- `context_limit`;
- `invalid_input`;
- `runtime_error`.

## Chat limitations

The local chat interface MUST display or return release limitations:

- the model was trained from a restricted corpus and may not know general facts;
- the model may be wrong or repetitive;
- the model may memorize training text;
- the example has no moderation service, human review queue, or production
  abuse monitoring;
- the model must not be used for medical, legal, financial, security-critical,
  or emergency decisions.

## Release transcript

Every release bundle MUST include a deterministic transcript generated from
the evaluation chat suite. The transcript MUST identify prompt ids, decoding
settings, checkpoint digest, and stop reasons. Transcript failures do not replace
the JSON evaluation report as the release source of truth.
