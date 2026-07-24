# Data Card: {{ corpus_name }}

## Summary

- Corpus owner: {{ corpus_owner }}
- Corpus version: {{ corpus_version }}
- Allowed use: {{ allowed_use }}
- Total declared items: {{ total_items }}
- Training items: {{ train_items }}
- Validation items: {{ validation_items }}
- Test items: {{ test_items }}
- Excluded items: {{ excluded_items }}

## Provenance and Licensing

Every included item must appear in the corpus manifest with provenance, license
classification, content hash, and review status.

| License class | Count | Notes |
|---|---:|---|
| {{ license_class }} | {{ license_count }} | {{ license_notes }} |

Items with unknown provenance or unknown license are excluded from target
training.

## Corpus Processing

- Normalization policy: {{ normalization_policy }}
- Split policy: conversation-level split before token packing
- Deduplication policy: {{ deduplication_policy }}
- Tokenizer training split: training only
- Restricted vocabulary size: {{ vocab_size }}
- Allowed character policy: {{ allowed_character_policy }}

## PII, Secret, and Safety Review

| Check | Count | Status |
|---|---:|---|
| PII candidates | {{ pii_candidate_count }} | {{ pii_status }} |
| Secret candidates | {{ secret_candidate_count }} | {{ secret_status }} |
| High-entropy strings | {{ high_entropy_count }} | {{ high_entropy_status }} |
| Prompt-injection candidates | {{ prompt_injection_count }} | {{ prompt_injection_status }} |
| Canary strings inserted | {{ canary_count }} | {{ canary_status }} |

Public data cards must not include raw private snippets, secrets, or personal
data.

## Coverage

Known covered domains:

{{ covered_domains }}

Languages:

{{ languages }}

Known excluded groups or topics:

{{ excluded_groups_or_topics }}

Evaluation category count gaps:

{{ evaluation_category_gaps }}

Known gaps and caveats:

{{ coverage_gaps }}

## Fingerprints

- Corpus manifest digest: {{ corpus_manifest_sha256 }}
- Raw corpus digest: {{ raw_corpus_sha256 }}
- Normalized corpus digest: {{ normalized_corpus_sha256 }}
- Split manifest digest: {{ split_manifest_sha256 }}
- Tokenizer digest: {{ tokenizer_sha256 }}
