.PHONY: install audit-data prepare-data model-plan tiny-smoke test lint typecheck

install:
	python -m pip install -e '.[dev]'

audit-data:
	python -m hello_slm.banking_data audit-sources

prepare-data:
	python -m hello_slm.banking_data prepare

model-plan:
	PYTHONPATH=src python scripts/banking_v2/train_banking_moe.py

tiny-smoke:
	PYTHONPATH=src python scripts/banking_v2/cloud_train_banking_moe.py \
		--run-tiny-smoke \
		--max-steps 1 \
		--output-dir artifacts/banking-v2-tiny-smoke

test:
	python -m pytest tests/test_banking_*.py poc/retail-bank-customer-service-poc/tests

lint:
	python -m ruff check src scripts tests poc/retail-bank-customer-service-poc

typecheck:
	python -m mypy src scripts tests
