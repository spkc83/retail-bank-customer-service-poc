# Tokenizer specification

This document is normative. `MUST`, `MUST NOT`, `SHOULD`, and `MAY` have their
usual requirements-language meanings.

## Scope

The tokenizer is trained from scratch on the configured restricted corpus. It
MUST NOT load pretrained vocabularies, merges, normalizers, or special tokens.
Tokenizer training uses only included manifest entries whose `allowed_use`
contains `tokenizer-training`.

## Algorithm

The reference tokenizer is byte-pair encoding over normalized Unicode text.
Training input is the rendered chat-template text from the training split only,
after data validation and normalization. Validation and test conversations MUST
NOT influence merge selection.

The tokenizer MUST be deterministic for the same:

- normalized training conversations;
- chat template version;
- special-token table;
- configured vocabulary size and frequency threshold;
- dependency versions;
- random seed, if the selected implementation uses one.

## Vocabulary budget

The total vocabulary size includes special tokens. All configured vocab sizes
MUST be `< 50000` for this example repository. The smoke tokenizer uses
`vocab_size = 256`.

For a focused small language model under 500M parameters, recommended configs
SHOULD keep vocabulary between `4096` and `16384` unless corpus analysis shows
that a different restricted vocabulary is needed.

## Immutable special tokens

Special token IDs are fixed and MUST NOT change across runs:

| Token | ID | Meaning |
|---|---:|---|
| `<|pad|>` | 0 | Padding only; excluded from loss. |
| `<|unk|>` | 1 | Unknown token for decode-time robustness only. |
| `<|bos|>` | 2 | Start of rendered conversation. |
| `<|eos|>` | 3 | End of rendered conversation. |
| `<|system|>` | 4 | System role marker. |
| `<|user|>` | 5 | User role marker. |
| `<|assistant|>` | 6 | Assistant role marker. |
| `<|end|>` | 7 | End of one message block. |

The first learned token ID MUST be `8`. A tokenizer artifact with different
special-token IDs MUST be rejected.

## Restricted vocabulary and unknown policy

Corpus validation rejects characters outside the configured allowlist before
tokenizer training. Therefore `<|unk|>` MUST NOT appear when encoding validated
training, validation, or test data. If any unknown token is produced while
building datasets, the stage MUST fail and report the conversation IDs.

Interactive chat may receive out-of-policy characters. The chat command MUST
apply the same normalization and then either:

- reject the input with a clear validation error when
  `interactive_unknown_policy = "reject"`; or
- encode unsupported spans as `<|unk|>` when explicitly configured for demos.

The default policy is `reject`.

## Normalization and pre-tokenization

The tokenizer MUST reuse the data spec's normalization. It MUST NOT introduce an
additional lossy normalizer.

Pre-tokenization for the reference implementation:

- Preserve the immutable special tokens as atomic tokens.
- Split ordinary text into runs of allowed letters/digits, runs of spaces,
  line feeds, and single punctuation characters.
- Learn BPE merges only inside ordinary text runs; never merge across special
  token boundaries or role/message block boundaries.

## Training order and tie-breaking

The BPE trainer processes conversations sorted by `conversation_id` after
normalization. Pair counts are aggregated over the rendered training text. Merge
selection order is:

1. Highest frequency.
2. Lexicographically smallest left token.
3. Lexicographically smallest right token.

Training stops when the vocabulary reaches `vocab_size` or no pair reaches
`min_pair_frequency`.

## Artifact requirements

The tokenizer artifact MUST include:

- `format_version = 1`;
- tokenizer algorithm and implementation version;
- immutable special token table;
- normalizer and pre-tokenizer identifiers;
- learned vocabulary and merges in deterministic order;
- source corpus manifest hash;
- canonical corpus fingerprint;
- tokenizer configuration hash;
- rendered training text hash;
- dependency versions;
- `tokenizer_fingerprint`.

`tokenizer_fingerprint` is SHA-256 over canonical JSON containing all behavior-
affecting tokenizer fields, excluding output path and creation timestamp.

## Encoding and decoding

Encoding MUST be deterministic and reversible for all text that passes the
configured allowlist, except for normalization-equivalent inputs that canonicalize
to the same NFC string. Decoding MUST preserve special tokens when requested and
MUST omit them only through an explicit `skip_special_tokens` option.

Padding uses `<|pad|>` and MUST be excluded from attention and loss. Sequence
packing MUST NOT insert padding inside a conversation. Truncation MUST occur only
at configured conversation boundaries; partial assistant targets are invalid for
evaluation examples.

## Acceptance metrics

Tokenizer validation passes only when all of the following are true:

- Special-token strings and IDs exactly match this specification.
- Vocabulary size is `<= vocab_size` and includes all special tokens.
- No validation corpus example encodes to `<|unk|>`.
- Re-training from the same config and corpus produces byte-identical tokenizer
  artifacts except for permitted timestamp fields.
- Encoding then decoding every smoke corpus message returns the normalized text.
- The tokenizer report includes character coverage, token counts per split,
  unknown-token counts, top tokens, merge count, compression ratio, and
  `tokenizer_fingerprint`.
