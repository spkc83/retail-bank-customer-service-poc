# Data and governance specification

This document is normative. `MUST`, `MUST NOT`, `SHOULD`, and `MAY` have their
usual requirements-language meanings.

## Scope

The corpus is a closed, declared set of JSONL conversation files. Every training,
validation, test, tokenizer-training, and evaluation row MUST be listed in the
configured corpus manifest and MUST pass `schemas/conversation.schema.json`.
Undeclared files, remote URLs, generated cache files, and hidden directory
contents MUST be ignored and reported as rejected input if discovered.

The hello-world corpus in `examples/corpus/` is synthetic and intentionally tiny.
It is only large enough to smoke-test validation, tokenizer training, dataset
packing, training, resume, evaluation, and bounded chat generation.

## Corpus manifest

The manifest MUST pass `schemas/corpus-manifest.schema.json`. It is the
governance source of truth for each corpus file.

Each entry MUST include:

- `path`: repository-relative path to one JSONL file.
- `split`: one of `train`, `validation`, or `test`.
- `sha256`: lowercase hex SHA-256 of the exact file bytes.
- `bytes`: exact byte count of the file.
- `conversation_count`: exact number of JSONL records.
- `provenance`: human-readable origin.
- `license`: license identifier or local grant name.
- `rights_holder`: owner or author of the text.
- `consent`: whether use for model training is permitted.
- `allowed_use`: list that MUST include `tokenizer-training` for files used to
  train the tokenizer and `model-training` for files used to train weights.
- `contains_personal_data`: boolean.
- `contains_synthetic_data`: boolean.
- `included`: boolean. Only included entries are consumed.

Entries with `included = false` MAY document rejected or quarantined files. They
MUST include `exclusion_reason`.

## JSONL conversation format

Each line is one complete conversation object. Blank lines are invalid. JSONL
records MUST use UTF-8 without a byte-order mark and MUST end with a newline.

Required top-level fields:

- `schema_version`: integer, currently `1`.
- `conversation_id`: stable unique string matching
  `^[a-z0-9][a-z0-9._-]{2,79}$`.
- `source`: short source name matching the manifest provenance.
- `license`: license or grant string matching the manifest license.
- `created_at`: ISO-8601 UTC timestamp.
- `messages`: ordered non-empty list of chat messages.

Optional top-level fields:

- `metadata`: object for non-training control data only. Values MUST NOT contain
  hidden prompt text, secrets, personal data, or labels that would be leaked into
  training unless explicitly rendered by a later stage.

Message fields:

- `role`: exactly `system`, `user`, or `assistant`.
- `content`: non-empty string after normalization.
- `loss`: boolean; default is `true` for assistant messages and `false` for
  system/user messages.

Role rules:

- A conversation MAY start with one `system` message.
- `system` MUST appear at most once and only as the first message.
- After an optional `system` message, roles MUST alternate `user`, `assistant`,
  `user`, `assistant`, and so on.
- The final message MUST be `assistant`.
- At least one `user` and one `assistant` message are required.
- Only assistant messages MAY set `loss = true`; system and user messages MUST
  have `loss = false` or omit the field.

## Normalization

Validation MUST normalize text before hashing the canonical corpus fingerprint,
tokenizer training, split assignment, deduplication, and packing.

Normalization steps, in order:

1. Decode as UTF-8 and reject invalid byte sequences.
2. Normalize all text fields to Unicode NFC.
3. Convert `\r\n` and `\r` inside message content to `\n`.
4. Replace tabs with one ASCII space.
5. Strip trailing spaces from each line.
6. Collapse runs of more than two blank lines inside a message to exactly two.
7. Reject messages that become empty.

The raw file SHA-256 in the manifest is computed before normalization. The
canonical corpus fingerprint is computed after normalization using canonical JSON
with sorted keys and UTF-8 encoding.

## Restricted character policy

The default policy is intentionally narrow for a restricted-vocabulary SLM.
After normalization, message content MUST contain only:

- Unicode general categories `Lu`, `Ll`, `Lt`, `Lm`, `Lo`, `Nd`, `Mn`, `Mc`,
  `Me`, `Pc`, `Pd`, `Po`, `Ps`, `Pe`, `Pi`, `Pf`, `Sc`, `Sk`, `Sm`, `Zs`,
  and line feed `\n` when the character is also present in the profile's
  explicit allowlist.
- ASCII control characters are forbidden except line feed.
- Private-use, surrogate, unassigned, bidirectional override, and zero-width
  characters are forbidden.

For the smoke corpus, `configs/corpus.toml` further restricts content to:

```text
A-Z a-z 0-9 space newline . , ? ! ' " : ; - ( ) /
```

Characters outside the configured allowlist MUST cause corpus validation to
fail. They MUST NOT be silently mapped, dropped, or tokenized as unknown text.

## Provenance, consent, and license gates

The validator MUST reject an included manifest entry when:

- `consent.training_allowed` is not `true`;
- `allowed_use` does not include the use required by the current stage;
- `license` is missing, unknown to the configured allowlist, or incompatible
  with model training;
- `contains_personal_data = true`;
- a source is marked as scraped, confidential, private, leaked, or third-party
  without an explicit training grant.

The synthetic example corpus uses the repository license and self-authored
training consent. It contains no personal data.

## Deduplication

Deduplication occurs after normalization and before split assignment checks.

Required exact duplicate checks in every mode:

- Exact conversation duplicate: identical canonical `messages` arrays.
- Exact assistant target duplicate: identical normalized assistant content.
- Exact question/prompt and assistant-target duplicates when a source-preparation
  pipeline exposes those fields.

`corpus.near_duplicate_mode` controls the additional similarity check:

- `exhaustive` compares normalized message text using Jaccard 3-gram similarity
  with a threshold of `>= 0.92`. This is intended for small audit corpora.
- `exact_only` skips pairwise similarity comparison for large training corpora
  where exhaustive comparison is computationally impractical. Exact duplicate
  and cross-split contamination checks remain mandatory, and the corpus report
  MUST state that similarity checking was not performed.

Exact duplicates MUST be rejected. In `exhaustive` mode, near duplicates across
different splits MUST be rejected. The smoke config uses `exhaustive`; the
prepared arithmetic corpus uses `exact_only` and performs exact question,
assistant-target, and pair deduplication during source preparation.

## Deterministic split policy

Splits are conversation-level. No message or packed token sequence may cross a
conversation split boundary.

The manifest declares the intended split for each file. The validator MUST also
compute a deterministic split for each conversation:

```text
split_key = sha256(str(seeds.corpus_split) || "\n" || conversation_id)
bucket = first_64_bits(split_key) / 2^64
```

The bucket is assigned using configured ratios. If a manifest file is split-
specific, every conversation in that file MUST match the file split unless
`enforce_manifest_split_only = true`. The smoke config sets
`enforce_manifest_split_only = true` because each split has a tiny hand-authored
fixture. Larger experiments SHOULD set it to `false`.

## Contamination controls

Before training artifacts are released, the pipeline MUST run contamination
checks between training data and all held-out evaluation prompts/responses.

Required checks:

- Exact normalized overlap between train and validation/test conversations.
- Exact assistant target overlap between train and validation/test.
- Near-duplicate overlap using the deduplication threshold when
  `near_duplicate_mode = "exhaustive"`.
- N-gram overlap report for 5-grams through 13-grams.

Any exact contamination is a hard failure. Similarity-detected contamination is
a hard failure when exhaustive checking is enabled. N-gram overlap above configured thresholds is a hard failure
unless the overlapping text is a declared fixed chat-template token sequence.

## Chat template

The rendered training text MUST use the tokenizer spec's immutable special
tokens exactly:

```text
<|bos|><|system|>
{system_content}<|end|>
<|user|>
{user_content}<|end|>
<|assistant|>
{assistant_content}<|end|><|eos|>
```

If no system message exists, rendering starts with `<|bos|><|user|>`. For
multi-turn conversations, user and assistant blocks repeat in order. The model
uses causal next-token prediction. Loss is masked for system/user content and
role markers. For an assistant message with `loss = true`, its content and
following `<|end|>` are loss-bearing; the final `<|eos|>` is also loss-bearing.
This teaches the model to terminate a response and a complete conversation.

## Acceptance metrics

Corpus validation passes only when all of the following are true:

- Manifest schema and every JSONL record schema pass.
- Every included file's byte count, SHA-256, and conversation count match.
- Every conversation ID is unique across the included corpus.
- Every record passes role, normalization, character, provenance, license, and
  consent rules.
- No exact duplicates exist; exhaustive mode additionally requires no cross-split
  near duplicates.
- No validation/test contamination exists.
- The generated corpus report includes per-split conversation counts, message
  counts, normalized character counts, rejected inputs, near-duplicate mode and
  coverage, and the canonical corpus fingerprint.
