# Verification plan

Verification is layered so a cheap failure prevents an expensive run. Evidence
from a target-scale run is never inferred from the smoke profile.

## Static gates

1. Parse every TOML/JSON/JSONL file.
2. Validate serialized records against their JSON schema.
3. Validate cross-file references and reject undeclared corpus files.
4. Compute the exact parameter count from the same model classes used by training.
5. Reject forbidden pretrained/download fields and any parameter count at or above
   500,000,000.
6. Check tokenizer special IDs, vocabulary cap, character policy, and artifact hash.
7. Check split isolation by conversation ID and normalized-content fingerprint.

## Unit tests

The suite MUST cover at least:

- accepted and rejected conversation records;
- undeclared files, bad hashes, duplicate IDs, invalid role order, empty content,
  disallowed characters, and the explicit `<unk>` behavior;
- deterministic BPE merges and stable encode/decode for the supported alphabet;
- immutable special-token IDs and maximum vocabulary size;
- causal attention and target alignment with padding ignored by loss;
- parameter estimator equality with an instantiated smoke model;
- the hard boundary at 499,999,999 versus 500,000,000 parameters;
- deterministic conversation splitting and no cross-split normalized duplicate;
- checkpoint round-trip and strict resume rejection on fingerprint change;
- bounded generation stopping at EOS or the token limit.

## Integration acceptance flow

CI MUST run the following logically equivalent sequence from a clean checkout:

```bash
hello-slm validate --config configs/smoke.toml
hello-slm build-tokenizer --config configs/smoke.toml
hello-slm build-dataset --config configs/smoke.toml
hello-slm train --config configs/smoke.toml --max-steps 2
hello-slm train --config configs/smoke.toml --resume artifacts/smoke/checkpoints/latest.pt --max-steps 3
hello-slm eval --config configs/smoke.toml --checkpoint artifacts/smoke/checkpoints/latest.pt
hello-slm chat --config configs/smoke.toml --checkpoint artifacts/smoke/checkpoints/latest.pt --prompt "hello" --max-new-tokens 4
```

The flow passes only when all commands return zero, resume advances the global
step without resetting optimizer/scheduler state, evaluation returns finite
metrics, and generation emits no token outside the stored vocabulary.

## Target-run gates

Before target training, archive:

- approved data card and immutable corpus snapshot;
- static validation report and parameter count;
- dependency lock, source commit, container/driver inventory, and hardware plan;
- tokenizer fertility/coverage report and manual sample audit;
- baseline and abort thresholds;
- checkpoint restore drill on representative hardware.

After target training, release requires every gate in `06-evaluation.md`, signed
hashes for the selected checkpoint and tokenizer, and completed model/data cards.

## Test environment

The smoke path MUST run on CPU with Python 3.11+ and PyTorch. CUDA MAY accelerate
it but MUST NOT be required. Tests MUST write to a temporary directory and MUST
NOT depend on network access, wall-clock timestamps, or ignored local artifacts.
