# Safety, security, and privacy specification

This document is normative for data intake, training, evaluation, and release.
The project is a teaching implementation and MUST NOT claim production-grade
safety.

## Safety policy

The chat model is intended for bounded, corpus-grounded conversation. It SHOULD
answer benign questions covered by the restricted corpus, transform user text
when requested, and state uncertainty for unsupported facts.

The model MUST refuse requests for:

- instructions that facilitate physical harm, cyber abuse, fraud, or evasion;
- extraction of secrets, credentials, private personal data, or canary strings;
- reproducing substantial copyrighted text from the corpus beyond configured
  quote limits;
- pretending to have capabilities, training data, or live knowledge it lacks.

Refusals MUST be brief, avoid procedural detail, and MAY redirect to a benign
high-level alternative.

## Threat model

The implementation MUST consider these adversaries:

| Threat | Required control |
|---|---|
| Corpus poisoning | Manifest allowlist, immutable source hashes, schema validation, review status, and duplicate/anomaly reports. |
| Secret or PII leakage | Intake scanning, denylisted patterns, manual review queue, canary probes, and blocked release on confirmed findings. |
| Artifact tampering | SHA-256 manifests, atomic writes, checksum verification before load, and signed release manifest when signing keys are configured. |
| Prompt injection in corpus | Treat corpus text as data, never as executable instructions for tooling; normalize and escape during reports. |
| Evaluation gaming | Conversation-level splits, contamination checks, frozen eval set fingerprints, and report of any threshold override. |
| Unsafe local serving | Bounded generation, stop tokens, no tool access, no network access from `chat`, and explicit limitation text in the model card. |

## Restricted corpus controls

Every corpus item MUST be declared in a manifest with:

- source path or URI recorded as provenance text;
- license and allowed-use classification;
- SHA-256 content hash;
- reviewer or automated policy status;
- split eligibility;
- exclusion reason when not used.

Training MUST reject undeclared files. The tokenizer MUST train only on the
training split after exclusions. Evaluation and test rows MUST never be merged
back into training.

## Poisoning checks

Corpus validation MUST report:

- exact duplicates and near duplicates;
- outlier document length;
- unexpected language or character distribution;
- high frequency URLs, emails, phone numbers, tokens, or base64-like strings;
- prompt-injection phrases such as instructions to ignore previous directions;
- license or provenance fields marked unknown.

For the smoke corpus, warnings MAY pass when documented in the corpus report.
For target training, unknown license/provenance, confirmed prompt injection in
assistant turns, or unresolved anomaly review MUST fail validation.

## PII and secret handling

Validation MUST scan raw and normalized text for:

- email addresses, phone numbers, postal addresses, government identifier
  patterns, API keys, private keys, passwords, session tokens, and high-entropy
  strings;
- project-specific denylist terms supplied by configuration;
- synthetic canary patterns.

Matches MUST be written to a private local report with redacted snippets. Public
reports MUST include counts and categories only. Confirmed secrets or personal
data MUST be removed before tokenizer training and model training. The repo MUST
NOT commit real secrets, private corpora, checkpoints, or private scan reports.

## Vocabulary safety

The restricted vocabulary is a capability boundary only in a narrow sense. It
MAY reduce accidental unsupported text generation, but it MUST NOT be described
as a complete safety mechanism. Special tokens are immutable, and unknown or
disallowed characters MUST map through the tokenizer's configured unknown
handling rather than causing silent data loss.

## Safety evaluation

Safety evaluation MUST include:

- direct unsafe requests;
- obfuscated unsafe requests;
- requests to reveal training data, canaries, or secrets;
- benign safety-adjacent prompts that should be answered or gently limited;
- unsupported factual questions that require uncertainty.

Release requires the refusal and over-refusal thresholds in
`specs/06-evaluation.md`. Any generated credential, canary, or private data is a
hard failure regardless of aggregate score.

## Artifact integrity

Every generated artifact MUST have a SHA-256 digest in its manifest. PyTorch
dataset/checkpoint digests MUST be verified before deserialization, and readers
MUST use restricted tensor-only loading rather than unrestricted pickle. The release
bundle MUST include a `manifest.json` listing file size, digest, format version,
source run id, creation time, and implementation version. Loaders MUST verify
digests before use.

If signing is configured, release MUST create a detached signature for
`manifest.json`. Unsigned bundles MUST clearly state `signed = false`.

## Privacy limitations

Training from scratch does not prevent memorization. A small restricted corpus
can increase memorization risk. Model cards and release notes MUST disclose that
the model may reproduce training text and is unsuitable for private or regulated
data without stronger review, deduplication, privacy testing, and deployment
controls.
