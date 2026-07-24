# Requirements traceability

This matrix is the completion ledger. File existence establishes `specified`;
the named tests and commands establish `verified` after implementation.

| Requirement | Primary specification | Implementation surface | Verification evidence |
|---|---|---|---|
| R-001 from-scratch training | `04-model.md`, `05-training.md` | `model.py`, `training.py` | model-init and checkpoint tests; smoke train command |
| R-002 `<500M` hard cap | `04-model.md` | `config.py`, `model.py` | boundary/estimator tests; validate both profiles |
| R-003 conversational focus | `02-data-and-governance.md`, `08-serving-and-chat.md` | `data.py`, `generation.py` | template/masking tests; eval/chat commands |
| R-004 restricted corpus | `02-data-and-governance.md` | `data.py` | manifest/hash/schema/undeclared-file tests |
| R-005 restricted vocabulary | `03-tokenizer.md` | `tokenizer.py` | special-ID, allowlist, unknown, round-trip tests |
| R-006 hello-world path | `10-verification-plan.md` | all CLI stages | clean end-to-end smoke test |
| R-007 focused target profile | `04-model.md`, `05-training.md` | `configs/focused-125m.toml` | structural validation and 129,000,192 parameter report |
| R-008 reproducibility | `05-training.md`, `09-reproducibility-and-operations.md` | `artifacts.py`, `training.py` | deterministic hash and strict resume tests |
| R-009 quality gates | `06-evaluation.md` | `evaluation.py` | finite held-out metrics and threshold report tests |
| R-010 safety/privacy/security | `07-safety-security-privacy.md` | validation and reports | PII/secret/character rejection tests; release remains blocked for smoke |
| R-011 build-ready interfaces | `01-system-architecture.md` plus schemas/configs | `cli.py` | CLI help and command integration tests |
| R-012 implementation after specs | entire numbered spec set | `src/hello_slm/`, `tests/` | version-control history/order plus full test suite |
| R-013 non-production boundary | `11-decisions-and-risks.md`, card templates | README and CLI output | documentation review; smoke reports `release_eligible=false` |
| R-014 arithmetic-tutor corpus | `02-data-and-governance.md`, `data/sources/orca-math.lock.json` | `prepare_math.py`, `arithmetic-30m.toml` | `test_prepare_math.py`; full arithmetic corpus validation |
| R-015 verified arithmetic curriculum | `03-tokenizer.md`, `06-evaluation.md` | `prepare_arithmetic_curriculum.py`, `arithmetic_evaluation.py`, `arithmetic-curriculum-30m.toml` | generator/tokenizer/evaluator tests; 2,000-step run; supported exact-answer gate |

## Verification commands

```bash
python -m pytest
python -m hello_slm validate --config configs/smoke.toml
python -m hello_slm validate --config configs/focused-125m.toml --structural
python -m hello_slm smoke --config configs/smoke.toml --work-dir <temporary-directory>
```

The implementation MAY expose the console-script equivalent `hello-slm`. The
module form is canonical for a checkout that has not been installed.
