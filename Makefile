.PHONY: install test lint type sec serve init schemas \
	train-install train-prepare train-baseline train-run train-all

install:
	py -m pip install -e .[dev]

test:
	pytest

lint:
	ruff check .
	ruff format --check .

type:
	mypy src

sec:
	bandit -q -r src

serve:
	aep serve

init:
	aep init

schemas:
	aep init --export-schemas

# ---------------------------------------------------------------------------
# Autoresearch training workflow
# ---------------------------------------------------------------------------

train-install:
	py -m pip install -e ".[train]"

train-prepare:
	aep research prepare --shards 100

train-baseline:
	aep research train --baseline --depth 8

train-run:
	aep research train --depth 8 --iterations 20

train-all: train-install train-prepare train-baseline train-run
	@echo "✓ Autoresearch training pipeline complete"
