.PHONY: install prepare-data model-plan tiny-smoke test lint typecheck

install:
	python -m pip install -e '.[dev]'

prepare-data:
	PYTHONPATH=src python scripts/retail_bank/prepare_tool_sft_data.py \
		--output-dir data/banking-v3-tool-sft \
		--pilot-count 5000

model-plan:
	PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
		--manifest data/banking-v3-tool-sft/manifest.json

tiny-smoke:
	PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
		--run-tiny-smoke \
		--family granite \
		--max-steps 1 \
		--output-dir artifacts/banking-v3-tool-sft-smoke

test:
	python -m pytest tests/test_banking_*.py poc/retail-bank-customer-service-poc/tests

lint:
	python -m ruff check src scripts tests poc/retail-bank-customer-service-poc

typecheck:
	python -m mypy src scripts tests
