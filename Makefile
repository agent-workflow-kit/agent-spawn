.PHONY: verify test lint fmt fmt-check typecheck mdlint

verify: lint fmt-check typecheck test mdlint

# Parallel by default (pytest-xdist); plain `pytest` stays serial. PYTEST_WORKERS=0 runs serial.
# `auto` follows the machine's CPU count, capped by PYTEST_MAXPROCS so a big machine is not flooded.
PYTEST_WORKERS ?= auto
PYTEST_MAXPROCS ?= 8

test:
	uv run pytest -n $(or $(PYTEST_WORKERS),auto) --maxprocesses=$(or $(PYTEST_MAXPROCS),8)

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

fmt-check:
	uv run ruff format --check .

typecheck:
	uv run pyright

mdlint:
	uv run pymarkdown --config .pymarkdown.json scan -r --respect-gitignore .
