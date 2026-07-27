# Banking dual-head domain and intent router

## Scope

The deployed router uses one shared DistilBERT encoder with two independently
calibrated outputs:

- a binary domain head for supported retail banking versus out-of-domain;
- a 77-way Banking77 intent head evaluated only for accepted banking inputs.

The learned router replaces keyword OOD classification in deployment. The
deterministic credential-input and unsafe-output guards remain separate safety
controls. If the learned router is missing, corrupt, or cannot load, serving
fails closed to the exact OOD response rather than falling back to keywords.

## Governed data

The router-training manifest is distinct from the generative SFT manifest.
Banking77 may be used for classifier training and evaluation but MUST NOT enter
generative SFT. Its original test split remains untouched. A deterministic
per-intent validation partition is carved only from its original training
split.

UCI CLINC150 supplies diverse domain negatives under CC-BY-4.0. CLINC labels
that overlap supported retail-banking capabilities are positive only for the
domain head and use intent label `-100`; they do not alter the Banking77 intent
ontology. Remaining CLINC intents and OOS examples are domain negatives.

Every prepared split records provenance, license, allowed use, SHA-256 digest,
PII scan count, cross-split duplicate count, and the audited CLINC label policy.
Context-transition examples are created within each split only:

- same-intent banking follow-up: banking domain with the Banking77 intent;
- previous banking plus current non-banking request: out-of-domain.

The rendered current request precedes previous context so a topic change cannot
be overridden by stale banking terms.

## Training and artifact contract

Training minimizes binary domain cross-entropy plus intent cross-entropy.
Intent loss ignores rows whose intent label is `-100`. The released artifact
contains:

- a standard Transformers encoder and tokenizer;
- `classifier_heads.safetensors`;
- `router_config.json` with intent labels and calibrated threshold;
- `metrics.json`;
- `manifest.json` with artifact sizes and SHA-256 digests;
- a model card with data attribution and limitations.

PyTorch pickle is prohibited. Serving verifies the artifact manifest before
loading tensors and does not require `trust_remote_code`.

## Release gates

The deployment threshold is selected on validation data with in-domain recall
of at least 98%. Deployment is blocked unless the untouched test split passes:

- Banking77 intent macro F1 at least 0.90;
- in-domain false-refusal rate at most 2%;
- OOD false-accept rate at most 5%;
- banking-follow-up false-refusal rate at most 5%;
- banking-to-OOD false-accept rate at most 5%;
- zero cross-split normalized duplicates;
- zero detected PII-like strings;
- every released artifact digest verified.

The public Space exposes route, calibrated banking probability, and predicted
intent through a route API. OOD requests bypass the generative model and return
the exact stock response.
