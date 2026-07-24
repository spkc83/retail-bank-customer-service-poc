# Architecture decisions and risk register

## Decisions

### ADR-001: Minimal custom PyTorch implementation

Use PyTorch primitives for the reference model and training loop. This keeps the
model definition, parameter count, checkpoint state, and causal masking visible.
Framework trainers were rejected for the hello-world path because their implicit
defaults obscure the normative contracts. A production implementation may adopt
one only after parity tests prove those contracts.

### ADR-002: Train a deterministic restricted BPE tokenizer

Word-only tokenization cannot express novel in-policy strings, while byte fallback
silently expands the allowed alphabet. A small BPE tokenizer over an explicit
normalized character allowlist provides bounded vocabulary and compositional
coverage. Characters outside policy follow the configured reject-or-`<unk>` rule;
there is no implicit byte fallback.

### ADR-003: Two configurations, one implementation

The smoke and focused profiles exercise identical code. The smoke profile proves
plumbing and resume behavior; the focused profile specifies a plausible model but
requires separate compute and quality evidence. Code MUST NOT special-case smoke
behavior except via configuration values.

### ADR-004: Conversation-level splits before tokenization

Splitting token windows leaks adjacent turns and near-duplicates. Stable hashing
of conversation IDs assigns whole conversations to a split before tokenizer
training or packing. Tokenizer training uses only the training split.

## Risk register

| Risk | Likelihood / impact | Control | Residual risk |
|---|---|---|---|
| Restricted corpus is too small or narrow for fluent chat. | High / high | Coverage report, staged corpus growth, held-out task gates. | A <500M model remains capacity-limited. |
| Memorization of sensitive text. | Medium / high | Rights review, PII/secret removal, canaries, extraction tests. | Automated detection has false negatives. |
| Train/eval contamination. | Medium / high | Split before packing, normalized dedup, task fingerprint checks. | Semantic duplicates may remain. |
| Vocabulary policy harms names or multilingual input. | High / medium | Publish supported alphabet/languages and explicit `<unk>` behavior. | User experience is intentionally constrained. |
| Synthetic smoke metrics are mistaken for quality. | Medium / high | Separate thresholds and conspicuous non-production labels. | Downstream users can ignore documentation. |
| Checkpoint cannot resume exactly across hardware. | Medium / medium | Persist all state; same-device restore drill; document nondeterminism. | Some GPU kernels are nondeterministic. |
| Poisoned or unlicensed corpus additions. | Medium / high | Manifest allowlist, hashes, provenance review, content scans. | Reviewers can approve bad sources. |
| Model emits unsafe or false content. | High / high | Scope-limited data, safety eval, bounded generation, application controls. | Training alone cannot guarantee safety. |
| Parameter estimator drifts from implementation. | Low / high | Instantiate smoke model in tests and compare exact counts. | Target instantiation may exceed test resources. |
| Artifact substitution. | Low / high | SHA-256 manifests, atomic writes, release verification. | SHA-256 does not authenticate an untrusted publisher. |

## Explicit non-goals

- Distributed training, elastic jobs, and cloud orchestration.
- Tool use, retrieval augmentation, multimodal input, or long-context serving.
- Production authentication, rate limiting, content moderation, or autoscaling.
- A claim that the example corpus produces a useful general assistant.
- Importing pretrained weights, embeddings, or tokenizers.
