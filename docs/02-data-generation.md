# Data Generation

The repository has two separate data lanes:

- Tool-use SFT data for the Granite generative agent.
- Classifier-only data for the dual-head router.

Do not mix the lanes. The generative SFT release is self-authored synthetic
data under MIT. Banking77 and CLINC are classifier/evaluation sources only and
do not contribute training rows to the generative SFT release.

## Tool-Use SFT Data

The tool-use dataset trains the 8.79B model to converse, ask clarifying
questions, call tools, consume tool results, and write final answers.

Main files:

| File | Purpose |
| --- | --- |
| [../src/hello_slm/banking_tool_sft_data.py](../src/hello_slm/banking_tool_sft_data.py) | Generates records, validates invariants, writes manifests and cards. |
| [../scripts/banking_v2/prepare_tool_sft_data.py](../scripts/banking_v2/prepare_tool_sft_data.py) | CLI wrapper around the generator. |
| [../poc/retail-bank-customer-service-poc/synthetic_bank.json](../poc/retail-bank-customer-service-poc/synthetic_bank.json) | Seed customer/account/card/transaction/transfer/service-case state. |
| [../data/banking-v3-tool-sft/manifest.json](../data/banking-v3-tool-sft/manifest.json) | Local generated split manifest. |
| [../data/banking-v3-tool-sft/DATA_CARD.md](../data/banking-v3-tool-sft/DATA_CARD.md) | Local generated dataset card. |
| [../data_cards/retail-bank-agent-sft.md](../data_cards/retail-bank-agent-sft.md) | Public dataset card. |

### Generated split counts

The active public data card reports:

| Split | Records |
| --- | ---: |
| Train | 6,304 |
| Validation | 1,349 |
| Frozen test | 1,347 |
| Total | 9,000 |

The corpus fingerprint is
`2bb7a400ed2556b15c7e5eb6147668041b5deef8ae4f037f9e2e52295ff29ab5`.
The released split seed is `711`, which is also the generator default.

### Scenario coverage

The local generated card records these scenario-family counts:

| Scenario family | Conversations |
| --- | ---: |
| `tool_success` | 3,006 |
| `no_tool_banking_faq` | 1,665 |
| `multi_turn` | 1,665 |
| `conversation` | 999 |
| `tool_error` | 666 |
| `clarification` | 333 |
| `hard_negative` | 333 |
| `ood` | 333 |

The generator covers all nine public tools:

- `list_accounts`
- `list_cards`
- `list_service_cases`
- `list_transactions`
- `list_transfers`
- `cancel_transfer`
- `dispute_transaction`
- `freeze_card`
- `replace_card`

The canonical public tool schema is produced by `public_tool_manifest()` in
[../src/hello_slm/banking_tool_sft_data.py](../src/hello_slm/banking_tool_sft_data.py).
The same schema shape is used by the POC in
[../poc/retail-bank-customer-service-poc/model_service.py](../poc/retail-bank-customer-service-poc/model_service.py).

### Record structure

Each record contains:

- `record_id`: stable record identifier;
- `schema_version`: `banking-tool-sft/v1`;
- `messages`: system, user, assistant, and tool messages;
- `expected.ordered_calls`: expected tool-call IDs in order;
- `expected.tool_calls`: expected public tool names and arguments;
- `expected.requires_tool`: whether the row requires tool execution;
- `expected.path`: response path such as tool success, clarification, FAQ, or OOD;
- `expected.grounding_facts`: facts the final response must preserve;
- `split_keys`: stable values used for deterministic split assignment;
- `provenance.source`: `self-authored-synthetic`.

Tool-bearing records include assistant tool-call messages followed by correlated
tool-result messages. System, user, and tool-result messages are context only.
Assistant tool calls and final assistant responses are trainable.

### Validation rules

`validate_records()` in
[../src/hello_slm/banking_tool_sft_data.py](../src/hello_slm/banking_tool_sft_data.py)
rejects records that violate release invariants, including:

- duplicate `record_id` values;
- unsupported provenance;
- duplicate normalized user text;
- unknown tools;
- unsupported tool arguments;
- missing or mismatched tool results;
- unstable tool-call IDs;
- semantically empty final responses;
- missing path-specific content for clarification, OOD, FAQ, and hard-negative
  rows.

The manifest validator, `validate_banking_tool_sft_manifest()`, re-reads split
files, checks record counts, and validates all records.

### Validate dataset generation

From the repository root:

```bash
PYTHONPATH=src python scripts/banking_v2/prepare_tool_sft_data.py \
  --output-dir /tmp/retail-bank-tool-sft-check \
  --pilot-count 9000 \
  --split-seed 711
```

The command writes:

- `/tmp/retail-bank-tool-sft-check/train.jsonl`
- `/tmp/retail-bank-tool-sft-check/validation.jsonl`
- `/tmp/retail-bank-tool-sft-check/test.jsonl`
- `/tmp/retail-bank-tool-sft-check/manifest.json`
- `/tmp/retail-bank-tool-sft-check/preparation-report.json`
- `/tmp/retail-bank-tool-sft-check/README.md`
- `/tmp/retail-bank-tool-sft-check/DATA_CARD.md`

Use a smaller pilot when testing the generator quickly:

```bash
PYTHONPATH=src python scripts/banking_v2/prepare_tool_sft_data.py \
  --output-dir /tmp/retail-bank-tool-sft-smoke \
  --pilot-count 1200
```

`--pilot-count` must be at least the number of required base scenarios. The
full release uses `9000`. Use `data/banking-v3-tool-sft` as the output
directory only when intentionally refreshing the repository's generated local
copy.

### Optional teacher wording pass

The generator can export teacher-realization requests:

```bash
PYTHONPATH=src python scripts/banking_v2/prepare_tool_sft_data.py \
  --output-dir /tmp/retail-bank-tool-sft-teacher-check \
  --pilot-count 1200 \
  --export-teacher-requests /tmp/teacher-requests.jsonl
```

Teacher responses are wording-only. `import_teacher_realizations()` proves that
tool calls, tool results, expected ordered calls, final state hashes, grounding
facts, and split keys are unchanged by checking immutable hashes before and
after applying teacher text.

The published 9,000-row release did not use a teacher realization pass. If you
experiment with one, the realizer requires an explicitly selected model and an
immutable revision:

```bash
PYTHONPATH=src python scripts/banking_v2/realize_tool_sft_teacher.py \
  --input-requests /tmp/teacher-requests.jsonl \
  --output-responses /tmp/teacher-responses.jsonl \
  --model MODEL_REPOSITORY \
  --revision IMMUTABLE_40_CHARACTER_REVISION
```

Do not treat teacher-realized output as the released corpus unless it passes
the same invariants and is published under a new dataset revision.

## Router Data

The router dataset trains the CPU classifier, not the generative agent.

Main files:

| File | Purpose |
| --- | --- |
| [../src/hello_slm/banking_router_data.py](../src/hello_slm/banking_router_data.py) | Builds in-domain/OOD and Banking77 intent examples. |
| [../scripts/banking_v2/prepare_dual_head_router_data.py](../scripts/banking_v2/prepare_dual_head_router_data.py) | Downloads pinned sources and writes governed splits. |
| [../data/banking-router-v1/manifest.json](../data/banking-router-v1/manifest.json) | Local generated router-data manifest. |
| [../data/sources/banking-router-v1.lock.json](../data/sources/banking-router-v1.lock.json) | Tracked release lock for split digests. |
| [../data_cards/retail-bank-router-training-data.md](../data_cards/retail-bank-router-training-data.md) | Public router dataset card. |

### Sources

The preparation script downloads:

- PolyAI Banking77 train and test CSVs from a pinned source repository
  revision.
- UCI CLINC150 `data_oos_plus.json` from the pinned archive member.

It verifies SHA-256 digests before writing any prepared splits. Banking77 rows
provide 77-way intent supervision. CLINC rows supervise the binary domain head;
their intent label is `-100`.

### Router split counts

The active router data card reports:

| Split | Rows |
| --- | ---: |
| Train | 44,432 |
| Validation | 8,589 |
| Test | 16,260 |

The prepared manifest records `pii_matches: 0` and `review_status:
automated-policy-pass`.

### Router input shape

Router examples use the text format from `render_router_input()` in
[../src/hello_slm/banking_router_data.py](../src/hello_slm/banking_router_data.py):

```text
[CURRENT]
<current user message>
[PREVIOUS_USER]
<optional previous user message>
```

The POC router adds recent assistant context for ambiguous short follow-ups in
[../poc/retail-bank-customer-service-poc/router.py](../poc/retail-bank-customer-service-poc/router.py).

### Validate router-data generation

From the repository root:

```bash
PYTHONPATH=src python scripts/banking_v2/prepare_dual_head_router_data.py \
  --output-dir /tmp/retail-bank-router-data-check \
  --source-lock /tmp/retail-bank-router-data-check/SOURCE_LOCK.json \
  --expected-release-lock data/sources/banking-router-v1.lock.json
```

The command writes:

- `/tmp/retail-bank-router-data-check/train.jsonl`
- `/tmp/retail-bank-router-data-check/validation.jsonl`
- `/tmp/retail-bank-router-data-check/test.jsonl`
- `/tmp/retail-bank-router-data-check/manifest.json`
- `/tmp/retail-bank-router-data-check/README.md`
- `/tmp/retail-bank-router-data-check/SOURCE_LOCK.json`

By default, the script compares produced split digests to the tracked release
lock. Use `data/banking-router-v1` as the output directory only when
intentionally refreshing the repository's generated local copy. Use
`--skip-release-digest-check` only for intentional experimental splits.

## Tests

Focused data tests live in:

- [../tests/test_banking_tool_sft_data.py](../tests/test_banking_tool_sft_data.py)
- [../tests/test_banking_tool_sft_release.py](../tests/test_banking_tool_sft_release.py)
- [../tests/test_banking_router_data.py](../tests/test_banking_router_data.py)
- [../tests/test_banking_router_preparation.py](../tests/test_banking_router_preparation.py)

Run them from the repository root:

```bash
python -m pytest -q \
  tests/test_banking_tool_sft_data.py \
  tests/test_banking_tool_sft_release.py \
  tests/test_banking_router_data.py \
  tests/test_banking_router_preparation.py
```
