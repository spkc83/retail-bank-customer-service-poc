# Dual-Head Banking77 and CLINC Router

This guide covers the active CPU router: governed Banking77 plus CLINC preparation, DistilBERT dual-head training, threshold calibration, publication, and serving behavior. The router does not select tools and does not supply tool arguments to the Granite model.

## Active Artifact IDs

| Artifact | Value | Owner |
| --- | --- | --- |
| Router repo | `spkc83/retail-bank-domain-intent-router` | [`src/hello_slm/banking_dual_head_router.py`](../src/hello_slm/banking_dual_head_router.py) |
| Router revision | `136ee159d19cda7f585dd122907bbeb1ef4ec4db` | [`src/hello_slm/banking_dual_head_router.py`](../src/hello_slm/banking_dual_head_router.py) |
| Router dataset repo | `spkc83/retail-bank-router-training-data` | [`scripts/retail_bank/train_dual_head_router.py`](../scripts/retail_bank/train_dual_head_router.py) |
| Router dataset revision | `54ff186a03501d76dc643dbed3d82729267ce811` | [`scripts/retail_bank/train_dual_head_router.py`](../scripts/retail_bank/train_dual_head_router.py) |
| Base encoder | `distilbert/distilbert-base-uncased` | [`scripts/retail_bank/train_dual_head_router.py`](../scripts/retail_bank/train_dual_head_router.py) |
| Base encoder revision | `12040accade4e8a0f71eabdb258fecc2e7e948be` | [`scripts/retail_bank/train_dual_head_router.py`](../scripts/retail_bank/train_dual_head_router.py) |

The public dataset card is [`data_cards/retail-bank-router-training-data.md`](../data_cards/retail-bank-router-training-data.md). The public model card is [`model_cards/retail-bank-domain-intent-router.md`](../model_cards/retail-bank-domain-intent-router.md).

## Architecture

[`scripts/retail_bank/train_dual_head_router.py`](../scripts/retail_bank/train_dual_head_router.py) trains one shared DistilBERT encoder with two heads:

- a binary domain head for supported retail banking vs out-of-domain;
- a 77-way Banking77 intent head.

The domain loss applies to every row. Intent loss ignores CLINC and conversational rows whose intent label is `-100`. Serving uses [`src/hello_slm/banking_dual_head_router.py`](../src/hello_slm/banking_dual_head_router.py), verifies the artifact manifest, loads `classifier_heads.safetensors`, and runs without `trust_remote_code`.

The runtime input format is:

```text
[CURRENT]
{current user turn}
[PREVIOUS_USER]
{previous user turn, only for contextual follow-ups}
```

[`LearnedBankingRouter.classify`](../src/hello_slm/banking_dual_head_router.py) first classifies the current user turn alone. It includes the previous user turn only if the current turn looks referential, for example with words such as `this`, `that`, `them`, `next`, or `again`.

## Data Preparation

The preparation script is [`scripts/retail_bank/prepare_dual_head_router_data.py`](../scripts/retail_bank/prepare_dual_head_router_data.py). It downloads:

| Source | Revision or checksum | Use |
| --- | --- | --- |
| PolyAI Banking77 train CSV | SHA-256 `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` | domain and intent supervision |
| PolyAI Banking77 test CSV | SHA-256 `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` | untouched test supervision |
| Banking77 release revision | `90d4e2ee5521c04fc1488f065b8b083658768c57` | provenance |
| Banking77 source revision | `57ec275d8078af65b7731c2a98be812d844a6d6b` | raw CSV URLs |
| CLINC150 ZIP | SHA-256 `0d8ecc3e1edd7b25cabde0177544ce536ddf773844bc80ef1a75f36e7f030ea2` | OOD and supported-conversation rows |
| CLINC member `clinc150_uci/data_oos_plus.json` | SHA-256 `bfcca9ae515623541dc1983c94c4ed7cae9d26b42ae47d74b972e51bb6f7a21f` | extracted CLINC payload |

Prepare and reproduce the released split digests:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_dual_head_router_data.py \
  --output-dir data/banking-router-v1 \
  --expected-release-lock data/sources/banking-router-v1.lock.json \
  --validation-fraction 0.15 \
  --seed 7101
```

The default command verifies the generated split SHA-256 values against [`data/sources/banking-router-v1.lock.json`](../data/sources/banking-router-v1.lock.json). Use `--skip-release-digest-check` only for an intentional experiment that must not be published over the active release.

Outputs:

- `data/banking-router-v1/train.jsonl`;
- `data/banking-router-v1/validation.jsonl`;
- `data/banking-router-v1/test.jsonl`;
- `data/banking-router-v1/manifest.json`;
- `data/banking-router-v1/README.md`;
- `data/banking-router-v1/SOURCE_LOCK.json` unless `--source-lock` points elsewhere.

The prepared public dataset contains 44,432 train rows, 8,589 validation rows, and 16,260 test rows. Banking77 is classifier-only and never enters the Granite generative SFT lane.

## Label Policy

[`src/hello_slm/banking_router_data.py`](../src/hello_slm/banking_router_data.py) owns the label mapping:

- Banking77 rows are in-domain and keep their 77-way intent labels.
- CLINC labels overlapping supported banking capabilities are in-domain with intent label `-100`.
- CLINC `greeting`, `thank_you`, `goodbye`, and `are_you_a_bot` are in-domain conversational rows with intent label `-100`.
- Other CLINC and OOS rows are out-of-domain.
- Same-intent follow-ups are built within each split.
- Banking-to-OOD transitions are built within each split.
- Cross-split normalized duplicates are removed.
- PII-like strings are counted and must be zero.

Tests for this policy live in [`tests/test_banking_router_data.py`](../tests/test_banking_router_data.py), [`tests/test_banking_router_preparation.py`](../tests/test_banking_router_preparation.py), and [`tests/test_banking_router_training.py`](../tests/test_banking_router_training.py).

## Training and Calibration

The trainer has no CLI flags. It is a publish path that requires `HF_TOKEN`.

```bash
HF_TOKEN=... uv run scripts/retail_bank/train_dual_head_router.py
```

Use a token with read access to `spkc83/retail-bank-router-training-data`, read access to `distilbert/distilbert-base-uncased`, and write access to `spkc83/retail-bank-domain-intent-router`. Do not commit or paste the token into scripts, docs, or shell history.

Training constants in [`scripts/retail_bank/train_dual_head_router.py`](../scripts/retail_bank/train_dual_head_router.py):

| Setting | Value |
| --- | --- |
| Seed | `7101` |
| Max length | `96` |
| Batch size | `64` |
| Epochs | `4` |
| Learning rate | `3e-5` |
| Weight decay | `0.01` |
| Warmup ratio | `0.10` |
| Intent loss weight | `0.7` |
| Conversational domain loss weight | `8.0` |

After each epoch, the script predicts on validation, calibrates the domain threshold, and scores the epoch. [`calibrate_threshold`](../scripts/retail_bank/train_dual_head_router.py) searches thresholds from `0.005` to `0.995` and selects the best specificity while enforcing:

- in-domain recall at least `0.98`;
- conversational in-domain recall at least `0.95`.

The selected epoch is then evaluated once on the untouched test split.

## Release Gates

[`release_gate_failures`](../scripts/retail_bank/train_dual_head_router.py) blocks publication unless the test metrics pass:

| Metric | Gate |
| --- | --- |
| Intent macro F1 | `>= 0.90` |
| In-domain false-refusal rate | `<= 0.02` |
| OOD false-accept rate | `<= 0.05` |
| Same-intent follow-up false-refusal rate | `<= 0.05` |
| Conversational false-refusal rate | `<= 0.05` |
| Banking-to-OOD transition false-accept rate | `<= 0.05` |

The released artifact reports:

| Metric | Value |
| --- | ---: |
| Intent macro F1 | `0.948425` |
| In-domain false-refusal rate | `0.005099` |
| OOD false-accept rate | `0.020109` |
| Follow-up false-refusal rate | `0.001623` |
| Conversational false-refusal rate | `0.050000` |
| Banking-to-OOD false-accept rate | `0.009783` |
| Calibrated lower boundary | `0.165000` |

## Publish Outputs

On success, [`publish_artifact`](../scripts/retail_bank/train_dual_head_router.py) uploads:

- standard Transformers encoder files;
- tokenizer files;
- `classifier_heads.safetensors`;
- `router_config.json`;
- `metrics.json`;
- `manifest.json`;
- `README.md`.

The artifact manifest lists file sizes and SHA-256 digests. [`verify_router_artifact`](../src/hello_slm/banking_dual_head_router.py) checks those digests before serving. PyTorch pickle files are not part of the release.

## Serving Boundaries

[`LearnedBankingRouter.from_hub`](../src/hello_slm/banking_dual_head_router.py) loads the pinned router revision from Hub. The POC uses two boundaries recorded in [`model_cards/retail-bank-domain-intent-router.md`](../model_cards/retail-bank-domain-intent-router.md):

- banking probability `< 0.165`: out-of-domain;
- banking probability `>= 0.50`: in-domain;
- middle range: uncertain and delegated to the Granite model.

The classifier's top intent predictions are diagnostics only. They do not enter the Granite prompt, select tools, or provide tool arguments. If an already-loaded router fails on a turn, the POC reports an uncertain route and delegates the turn to the model; it does not silently substitute a keyword classifier.

## Stop Conditions

Stop before publication if:

- source digests do not match;
- generated split digests drift from [`data/sources/banking-router-v1.lock.json`](../data/sources/banking-router-v1.lock.json);
- cross-split duplicates or PII-like matches are nonzero;
- `HF_TOKEN` is unavailable for publish training;
- any release gate fails;
- the artifact manifest cannot verify every file;
- serving requires `trust_remote_code`.

Run the focused tests after any router change:

```bash
python -m pytest -q tests/test_banking_router_data.py \
  tests/test_banking_router_preparation.py \
  tests/test_banking_router_training.py \
  tests/test_banking_dual_head_router.py
```
