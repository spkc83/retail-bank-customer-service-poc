# Banking-v2 specification

Banking-v2 is a separate retail-banking adaptation track. It does not replace the
arithmetic hello-world model, does not use the v1 corpus manifest contract, and
does not claim that 24k SFT examples are sufficient for 9B pretraining.

## Source roles

- `bitext/Bitext-retail-banking-llm-chatbot-training-dataset`
  - License: `CDLA-Sharing-1.0`.
  - Role: primary generative QA SFT.
  - Revision: `3e3621092fc6baaf7f53ceb6f091c60ae99acb67`.
- `PolyAI/banking77`
  - License: `CC-BY-4.0`.
  - Role: intent-router and evaluation only.
  - Revision: `90d4e2ee5521c04fc1488f065b8b083658768c57`.
  - MUST NOT enter generative SFT files.
- Talkmap
  - Quarantined metadata only.
  - MUST NOT be acquired or emitted as trainable data.
- Rakesh sources
  - Excluded.
  - MUST NOT be acquired or emitted.

## Manifest contract

Banking-v2 uses `format_version = 2` with `contract =
"banking-v2-manifest"`. The manifest has two top-level lanes:

- `generative_sft`: train, validation, and test JSONL for chat adaptation.
- `router_eval`: Banking77 intent-router/eval JSONL, excluded from generative SFT.

The v1 `load_and_validate_corpus` path is intentionally not claimed compatible:
v1 rejects duplicate assistant targets and only allowlists MIT. Banking-v2 needs
repeated exact OOD responses for `task = "ood_gate"` and preserves Bitext and
Banking77 license attribution.

## Transform rules

- Source acquisition is separate from transformation.
- Local source snapshots and fingerprints are written under `data/sources/`.
- Bitext placeholders of the form `{{...}}`, malformed `{{...`, and malformed
  `...}}` are normalized to generic non-PII banking text.
- Any unresolved `{{` or `}}` after normalization is a hard failure.
- Metadata MUST include source dataset, revision, license, category, intent,
  task, trainability, split group, and placeholder replacement count where
  applicable.

## Split and dedup rules

- Bitext rows are deduplicated by normalized user instruction.
- Bitext rows are clustered inside each `(category, intent)` group with
  normalized token 1- to 3-gram Jaccard union-find.
- Threshold: `0.10`.
- Split key: `category:intent:cluster`.
- All rows in a cluster MUST remain in one split.
- Normalized user text MUST NOT appear across more than one split.

## OOD and multi-turn rules

- Out-of-domain prompts use the exact response:

```text
I can only help with retail banking and financial-services questions. Please ask about accounts, cards, transfers, payments, loans, or related banking support.
```

- Repeated exact assistant targets are allowed only when `metadata.task =
  "ood_gate"`.
- Multi-turn records MUST include clarification, follow-up, correction, and
  in-domain-to-OOD transition patterns.
- In-domain-to-OOD transitions that emit the canned response MUST use
  `task = "ood_gate"`.

## Acceptance gates

- Zero unresolved placeholders.
- Zero detected email, phone-number, payment-card, or account-number-like strings
  after the deterministic PII scrubber.
- Zero normalized user-text cross-split duplicates.
- Zero Banking77 rows in generative SFT.
- Zero trainable quarantined rows.
- Banking77 router/eval rows are marked `trainable = false`.
- OOD canned-response duplicates occur only under `task = "ood_gate"`.
- Every train, validation, and test split includes OOD examples and all four
  required multi-turn patterns.
- Generated report includes counts, licenses, source roles, fingerprints, split
  counts, and known integration gaps.

## Model and serving specifications

- [Model conversion and training](02-model-training.md)
- [Evaluation and serving](03-evaluation-serving.md)

`data/banking-v2/` and downloaded source snapshots are generated local data and
are ignored by Git. The source lock file remains tracked.
