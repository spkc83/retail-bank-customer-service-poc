.PHONY: install prepare-arithmetic prepare-curriculum validate validate-arithmetic validate-curriculum test lint smoke clean-artifacts

install:
	python -m pip install -e '.[dev]'

validate:
	python -m hello_slm validate --config configs/smoke.toml
	python -m hello_slm validate --config configs/focused-125m.toml --structural
	python -m hello_slm validate --config configs/arithmetic-30m.toml --structural
	python -m hello_slm validate --config configs/arithmetic-curriculum-30m.toml --structural

prepare-arithmetic:
	python -m hello_slm.prepare_math

prepare-curriculum:
	python -m hello_slm.prepare_arithmetic_curriculum

validate-arithmetic:
	python -m hello_slm validate --config configs/arithmetic-30m.toml

validate-curriculum:
	python -m hello_slm validate --config configs/arithmetic-curriculum-30m.toml

test:
	python -m pytest

lint:
	python -m ruff check src tests

smoke:
	python -m hello_slm smoke --config configs/smoke.toml

clean-artifacts:
	@echo "Remove artifacts/ manually after confirming no checkpoint is needed."
