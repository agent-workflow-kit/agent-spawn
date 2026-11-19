.PHONY: verify test live live-list lint fmt fmt-check typecheck mdlint

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

# The live suite runs the real Claude Code (haiku, tiny prompts), tmux, claude-session, termctl and
# wtctl. It is opt-in (AS_LIVE=1), never part of `verify` or CI, and billed scenarios need
# credentials: see docs/live-testing.md. Select with LIVE_SCENARIOS (scenario names, wave:<name>,
# billed, unbilled) or PYTEST_ARGS='-k ...'; cap Claude launches with MAX_LIVE_SESSIONS (default 25).
# Do not run it under pytest-xdist: the session counter and the wall-clock bounds are per process.
live:
	AS_LIVE=1 uv run pytest -m live tests/live -rs $(PYTEST_ARGS)

# Lists the scenarios and their expected billed sessions and turns; starts nothing.
live-list:
	uv run python -m tests.live
