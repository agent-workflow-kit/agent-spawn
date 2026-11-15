"""`live` tests are deselected by default and never run in CI."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items() if k not in ("CI", "LIVE_TMUX")}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=REPO_ROOT,
        env={**base, **(env or {})},
        capture_output=True,
        text=True,
        timeout=300,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def test_the_marker_is_registered_and_deselected_by_default() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["pytest"][
        "ini_options"
    ]
    assert any(m.startswith("live:") for m in config["markers"])
    assert "-m 'not live'" in config["addopts"]


def test_plain_pytest_deselects_every_live_test() -> None:
    done = pytest_run("tests/live", "--collect-only", "-q")
    assert "deselected" in done.stdout, done.stdout + done.stderr
    assert "test_fidelity.py::" not in done.stdout  # nothing selected


def test_live_tests_are_skipped_in_ci_even_when_asked_for() -> None:
    done = pytest_run("tests/live", "-m", "live", "-q", "-rs", env={"CI": "true"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "live tests never run in CI" in done.stdout
    assert " passed" not in done.stdout and " failed" not in done.stdout
