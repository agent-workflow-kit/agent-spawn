"""The fakes agree with the shared table of argv cases (the real programs agree in `live`)."""

from __future__ import annotations

import json

import pytest

from tests.fakebin.fidelity_cases import CASES
from tests.fakebin.harness import Fakebin

PARAMS = [
    pytest.param(tool, argv, exit_code, error_code, id=f"{tool}:{' '.join(argv)}")
    for tool, rows in CASES.items()
    for argv, exit_code, error_code, _ in rows
]


@pytest.mark.parametrize(("tool", "argv", "exit_code", "error_code"), PARAMS)
def test_the_fake_gives_the_exit_and_error_code_the_table_records(
    fakebin: Fakebin, tool: str, argv: list[str], exit_code: int, error_code: str | None
) -> None:
    done = fakebin.run(tool, *argv)
    assert done.returncode == exit_code, (done.stdout, done.stderr)
    code: str | None = None
    try:
        code = json.loads(done.stdout)["error"]["code"]
    except (ValueError, KeyError, TypeError):
        pass
    assert code == error_code


def test_every_tool_has_cases_and_most_are_verified_against_the_real_program() -> None:
    assert set(CASES) == {"claude-session", "termctl", "wtctl"}
    for tool in ("claude-session", "termctl"):
        verified = [row for row in CASES[tool] if row[3]]
        assert len(verified) >= 20
