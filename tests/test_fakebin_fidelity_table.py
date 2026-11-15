"""The fakes agree with the shared table of argv cases (the real programs agree in `live`)."""

from __future__ import annotations

import json

import pytest

from tests.fakebin.fidelity_cases import ACCEPTED, CASES, DIVERGENCES
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


@pytest.mark.parametrize(
    ("tool", "argv"),
    [pytest.param(t, a, id=f"{t}:{' '.join(a)}") for t, rows in ACCEPTED.items() for a in rows],
)
def test_the_fake_accepts_the_argv_the_real_program_accepts(
    fakebin: Fakebin, tool: str, argv: list[str]
) -> None:
    done = fakebin.run(tool, *argv)
    assert done.returncode != 2, (done.stdout, done.stderr)


@pytest.mark.parametrize(
    ("tool", "argv", "real_exit", "fake_exit"),
    [pytest.param(t, a, r, f, id=f"{t}:{' '.join(a)}") for t, a, r, f in DIVERGENCES],
)
def test_a_known_divergence_gives_the_fakes_exit(
    fakebin: Fakebin, tool: str, argv: list[str], real_exit: int, fake_exit: int
) -> None:
    assert real_exit != fake_exit
    assert fakebin.run(tool, *argv).returncode == fake_exit


def test_every_tool_has_cases_and_most_are_verified_against_the_real_program() -> None:
    assert set(CASES) == {"claude-session", "termctl", "wtctl"}
    for tool in ("claude-session", "termctl"):
        verified = [row for row in CASES[tool] if row[3]]
        assert len(verified) >= 20
