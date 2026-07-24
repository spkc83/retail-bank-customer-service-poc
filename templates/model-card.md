# Model Card: {{ model_name }}

## Summary

- Model type: decoder-only Transformer language model
- Parameter count: {{ parameter_count }}
- Training source: initialized from random weights and trained only on the
  declared restricted corpus
- Vocabulary size: {{ vocab_size }}
- Intended use: bounded local conversational demo for the covered corpus
- Release eligibility: {{ release_eligible }}

## Intended Use

This model is intended for short chat interactions within the topics and styles
represented by the restricted corpus. It is not a general assistant.

Do not use this model for medical, legal, financial, emergency, security, or
other high-impact decisions.

## Training Data

Reference the accompanying data card for corpus provenance, license classes,
exclusions, split counts, tokenizer policy, PII/secret scan status, canary
configuration, and known coverage gaps.

## Evaluation

| Metric | Value | Threshold | Pass |
|---|---:|---:|---|
| Held-out perplexity | {{ heldout_perplexity }} | {{ heldout_perplexity_threshold }} | {{ heldout_perplexity_pass }} |
| Chat exact match | {{ chat_exact_match }} | {{ chat_exact_match_threshold }} | {{ chat_exact_match_pass }} |
| Chat keyword recall | {{ chat_keyword_recall }} | {{ chat_keyword_recall_threshold }} | {{ chat_keyword_recall_pass }} |
| Format pass rate | {{ format_pass_rate }} | {{ format_pass_rate_threshold }} | {{ format_pass_rate_pass }} |
| Refusal pass rate | {{ refusal_pass_rate }} | {{ refusal_pass_rate_threshold }} | {{ refusal_pass_rate_pass }} |
| Over-refusal rate | {{ over_refusal_rate }} | {{ over_refusal_rate_threshold }} | {{ over_refusal_rate_pass }} |
| Canary extraction rate | {{ canary_extraction_rate }} | 0.0 | {{ canary_extraction_pass }} |
| Contamination rate | {{ contamination_rate }} | 0.0 | {{ contamination_pass }} |
| Distinct-2 | {{ distinct_2 }} | {{ distinct_2_threshold }} | {{ distinct_2_pass }} |

## Safety and Privacy

The model was evaluated for refusal behavior, over-refusal, contamination, and
canary extraction. These checks are limited and do not prove production safety.

Known limitations:

- may generate incorrect or unsupported answers;
- may repeat or memorize training text;
- may reflect bias and omissions in the restricted corpus;
- has no hosted moderation, abuse monitoring, or human escalation workflow;
- has no formal privacy guarantee.

## Reproducibility

- Run id: {{ run_id }}
- Effective config digest: {{ config_sha256 }}
- Corpus digest: {{ corpus_sha256 }}
- Tokenizer digest: {{ tokenizer_sha256 }}
- Dataset digest: {{ dataset_sha256 }}
- Checkpoint digest: {{ checkpoint_sha256 }}
- Evaluation report digest: {{ eval_report_sha256 }}

## Release Decision

Release decision: {{ release_decision }}

Rationale:

{{ release_rationale }}
