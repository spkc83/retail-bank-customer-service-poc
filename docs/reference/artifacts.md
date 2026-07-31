# Artifact Ledger

This page records the active immutable artifacts for the retail-bank Granite
agent, dual-head router, datasets, local manifests, and paid job outputs.

## Published Repositories

| Artifact | Repository | Immutable revision | Source |
| --- | --- | --- | --- |
| Granite retail-bank agent | `spkc83/retail-bank-agent-9b` | `085df3d089cfadd77424b548542da0390a54a23e` | [`model_cards/retail-bank-agent-9b.md`](../../model_cards/retail-bank-agent-9b.md), [`zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| Agent training/provenance revision | `spkc83/retail-bank-agent-9b` | `247ac402989144698f89727a59a07ce5d05f31c6` | [`model_cards/retail-bank-agent-9b.md`](../../model_cards/retail-bank-agent-9b.md) |
| Agent published evaluation head | `spkc83/retail-bank-agent-9b` | `98cde9ee058b785fb871abcd2c85e18cea410bdf` | [`model_cards/retail-bank-agent-9b.md`](../../model_cards/retail-bank-agent-9b.md) |
| Agent base model | `ibm-granite/granite-4.1-8b` | `1504002f650e656a0a3789d99574df12e3e94ed0` | [`configs/banking-tool-sft-granite.toml`](../../configs/banking-tool-sft-granite.toml), [`model_cards/retail-bank-agent-9b.md`](../../model_cards/retail-bank-agent-9b.md) |
| Tool-use SFT dataset | `spkc83/retail-bank-agent-sft` | `183e7e1ed1aba9c3d7155e7b83b64dc854935055` | [`data_cards/retail-bank-agent-sft.md`](../../data_cards/retail-bank-agent-sft.md), [`model_cards/retail-bank-agent-9b.md`](../../model_cards/retail-bank-agent-9b.md) |
| Dual-head router | `spkc83/retail-bank-domain-intent-router` | `136ee159d19cda7f585dd122907bbeb1ef4ec4db` | [`router.py`](../../poc/retail-bank-customer-service-poc/router.py), [`model_cards/retail-bank-domain-intent-router.md`](../../model_cards/retail-bank-domain-intent-router.md) |
| Router dataset | `spkc83/retail-bank-router-training-data` | `54ff186a03501d76dc643dbed3d82729267ce811` | [`train_dual_head_router.py`](../../scripts/retail_bank/train_dual_head_router.py), [`model_cards/retail-bank-domain-intent-router.md`](../../model_cards/retail-bank-domain-intent-router.md) |
| Public Space | `spkc83/retail-bank-servicing-poc` | Space commit is exposed at runtime as `SPACE_COMMIT_SHA` | [`app.py`](../../poc/retail-bank-customer-service-poc/app.py), [`README.md`](../../poc/retail-bank-customer-service-poc/README.md) |

## Agent Model Details

| Field | Value | Source |
| --- | --- | --- |
| Model repository | `spkc83/retail-bank-agent-9b` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Immutable weights revision | `085df3d089cfadd77424b548542da0390a54a23e` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Training/provenance revision | `247ac402989144698f89727a59a07ce5d05f31c6` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Published evaluation head | `98cde9ee058b785fb871abcd2c85e18cea410bdf` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Base model | `ibm-granite/granite-4.1-8b` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Base revision | `1504002f650e656a0a3789d99574df12e3e94ed0` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Source revision | `4270636255515f7a563d935794a3642e0b13ccb3` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Recovery source revision | `0237b97c0a9558bbb2e95c45097ac5ae5f9f7f21` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Dataset revision | `183e7e1ed1aba9c3d7155e7b83b64dc854935055` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Dataset fingerprint | `2bb7a400ed2556b15c7e5eb6147668041b5deef8ae4f037f9e2e52295ff29ab5` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Parameters | 8,791,592,960 | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Tool format | Granite native tagged JSON | [`model card`](../../model_cards/retail-bank-agent-9b.md), [`model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) |

## Paid Job Records

These are the job records for the released artifacts. Do not start new paid jobs unless
explicitly authorized.

| Purpose | Job ID | Evidence |
| --- | --- | --- |
| Granite SFT training | `spkc83/6a6a60d4b36a6516e96a0709` | [`model_cards/retail-bank-agent-9b.md`](../../model_cards/retail-bank-agent-9b.md) |
| FP16-native recovery and merge parity | `spkc83/6a6a6b6323ed89c748ec502c` | [`model_cards/retail-bank-agent-9b.md`](../../model_cards/retail-bank-agent-9b.md) |
| Frozen tool/final-response evaluation | `spkc83/6a6a6c7cb36a6516e96a0ac4` | [`model_cards/retail-bank-agent-9b.md`](../../model_cards/retail-bank-agent-9b.md) |

## Private Job-Bucket Retention

Bucket: `spkc83/jobs-artifacts`

The bucket is a private, restartable job workspace. It is not read by the
public Space and it is not the authoritative location for published model,
router, dataset, or evaluation artifacts.

| State | Files | Logical size |
| --- | ---: | ---: |
| Before 2026-07-31 cleanup | 290 | 449,461,595,301 bytes |
| After cleanup | 58 | 1,252,559,272 bytes |
| Removed | 232 | about 448.2 GB |

The retained set preserves the released continuation's selected
`checkpoint-600` adapter, trainer state, and provenance JSON. Failed runs,
superseded checkpoints, duplicate merged weights, temporary merge files, and
bucket copies of already-published evaluation outputs were removed. New runs
must write a new prefix and apply the same publish-verify-retain policy.

## Released Evaluation

The released evaluation report is stored under
`evaluation/085df3d089cf-183e7e1ed1ab/` in the model repository.

| Metric | Result | Source |
| --- | ---: | --- |
| Frozen test conversations | 1,347 | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Tool names and arguments | `774/774` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Executable trajectories | `678/678` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Dependent multi-tool sequences | `96/96` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Clarifications | `63/63` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Banking FAQ answers | `258/258` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| OOD paths | `30/30` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Grounded factual responses | `1,119/1,119` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Malformed calls, unsupported/private arguments, credential requests, in-domain false refusals, OOD false accepts | `0` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |

## Tool-Use SFT Dataset

Published repository: `spkc83/retail-bank-agent-sft`

Published revision:
`183e7e1ed1aba9c3d7155e7b83b64dc854935055`

Corpus fingerprint:
`2bb7a400ed2556b15c7e5eb6147668041b5deef8ae4f037f9e2e52295ff29ab5`

Local manifest:
[`data/banking-v3-tool-sft/manifest.json`](../../data/banking-v3-tool-sft/manifest.json)

| Split | Records | Local SHA-256 |
| --- | ---: | --- |
| train | 6,304 | `8d92fa0ab1d39875f0c4d918bc5aeaf670f71bf660a01a0a376cb4edc1cced53` |
| validation | 1,349 | `a8c7871b33689fce026ea570ad0a8a90a609cde232a89486e5437b028279e6d3` |
| test | 1,347 | `76b485fa507d56002f12b556f100fd842c77146804cf49be3426be031cc692c0` |

Tool manifest hash:
`sha256:88b6f53e19779732cde99190ebb5405d317e5691f91bc63624ef32c241939b40`

Coverage:

| Scenario family | Conversations |
| --- | ---: |
| Clarification | 333 |
| Conversation | 999 |
| Hard negative | 333 |
| Multi-turn | 1,665 |
| No-tool banking FAQ | 1,665 |
| OOD | 333 |
| Tool error | 666 |
| Tool success | 3,006 |

## Router Artifact

Published repository: `spkc83/retail-bank-domain-intent-router`

Published revision:
`136ee159d19cda7f585dd122907bbeb1ef4ec4db`

Training-data revision:
`54ff186a03501d76dc643dbed3d82729267ce811`

Router code:
[`poc/retail-bank-customer-service-poc/router.py`](../../poc/retail-bank-customer-service-poc/router.py)

| Field | Value |
| --- | ---: |
| Intent macro F1 | `0.948425` |
| In-domain false-refusal rate | `0.005099` |
| OOD false-accept rate | `0.020109` |
| Follow-up false-refusal rate | `0.001623` |
| Conversational false-refusal rate | `0.050000` |
| Banking-to-OOD false-accept rate | `0.009783` |
| Calibrated lower boundary | `0.165000` |
| Serving in-domain boundary | `0.50` |

Serving routes:

- banking probability `< 0.165`: `out_of_domain`
- banking probability `>= 0.50`: `in_domain`
- banking probability from `0.165` through `< 0.50`: `uncertain`

## Router Dataset

Published repository: `spkc83/retail-bank-router-training-data`

Published revision:
`54ff186a03501d76dc643dbed3d82729267ce811`

Local manifest:
[`data/banking-router-v1/manifest.json`](../../data/banking-router-v1/manifest.json)

Release lock:
[`data/sources/banking-router-v1.lock.json`](../../data/sources/banking-router-v1.lock.json)

| Split | Rows | Local SHA-256 |
| --- | ---: | --- |
| train | 44,432 | `c9067a04cefa90ed6fd874a6ebabbccb8015122b7920608c6d051df77b9f1acd` |
| validation | 8,589 | `7c1315a5f8555168143a0800364a98a6ce4f88ec7cb15b42216b15a83f33cfc5` |
| test | 16,260 | `e9178afc36ac2d90eef0587b4d42e2785e1c12e0a23445afe56486c3f61ac431` |

Prepared manifest SHA-256:
`78ba999216f0058d70db810c79bb4318f34c077bc81adc3adfae604d82c207f7`

Source locks:

| Source | Revision or digest |
| --- | --- |
| Banking77 release revision | `90d4e2ee5521c04fc1488f065b8b083658768c57` |
| Banking77 source repository revision | `57ec275d8078af65b7731c2a98be812d844a6d6b` |
| Banking77 normalized snapshot SHA-256 | `22ce056724069f431b477aa8478f1a42ce31286ad595cb7e53a838173052b340` |
| CLINC150 archive SHA-256 | `0d8ecc3e1edd7b25cabde0177544ce536ddf773844bc80ef1a75f36e7f030ea2` |
| CLINC150 member SHA-256 | `bfcca9ae515623541dc1983c94c4ed7cae9d26b42ae47d74b972e51bb6f7a21f` |

## Runtime Artifact Defaults

| Runtime field | Default | Source |
| --- | --- | --- |
| `RETAIL_BANK_MODEL_ID` | `spkc83/retail-bank-agent-9b` | [`zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| `RETAIL_BANK_MODEL_REVISION` | `085df3d089cfadd77424b548542da0390a54a23e` | [`zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| `ROUTER_REPO_ID` | `spkc83/retail-bank-domain-intent-router` | [`router.py`](../../poc/retail-bank-customer-service-poc/router.py) |
| `ROUTER_REVISION` | `136ee159d19cda7f585dd122907bbeb1ef4ec4db` | [`router.py`](../../poc/retail-bank-customer-service-poc/router.py) |
| `INPUT_TOKEN_BUDGET` | `8192` | [`model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) |
| `MAX_NEW_TOKENS` | `512` | [`model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) |
| `MAX_TOOL_CALLS` | `8` | [`model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) |
| Demo usernames | `alex.demo`, `maya.demo` | [`auth.py`](../../poc/retail-bank-customer-service-poc/auth.py) |
| Session database directory | `/tmp/retail-bank-servicing-poc` unless `POC_SESSION_DB_DIR` is set | [`state.py`](../../poc/retail-bank-customer-service-poc/state.py) |

## Source Commit For A New Run

The released job source revisions are recorded above. For a new paid run,
commit and push the intended source state, then obtain its immutable revision
with:

```bash
git rev-parse HEAD
```

Paid launchers require that value to be an exact 40-character commit and verify
that the remote job script exists at the same revision before starting work.
