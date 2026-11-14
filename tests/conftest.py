"""Shared fixtures: environment isolation, the fake binaries and a private tmux server."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests import isolation
from tests.fakebin.harness import Fakebin
from tests.private_tmux import PrivateTmux

pytest_plugins = ["pytester"]

#: The real tmux, found before any test changes ``PATH`` (the fakes shadow it later).
REAL_TMUX = shutil.which("tmux")


@dataclass(frozen=True)
class Isolation:
    """The per-test home and XDG directories (all temporary) and the subprocess environment."""

    home: Path
    config: Path
    state: Path
    data: Path
    cache: Path
    tmux_tmpdir: Path
    env: dict[str, str]


@pytest.fixture(autouse=True)
def isolation_env(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Isolation]:
    """Point HOME and the XDG variables at temporary directories for every test.

    Also drops variables that would reach the user's tmux server or another tool's state
    (``TMUX``, ``AGENT_SPAWN_*``, ...). The guard fails the test, at setup and again at
    teardown, if the environment names a real directory of the user.
    """
    # Not tmp_path: tests assert that the directory they were given stays empty.
    root = tmp_path_factory.mktemp("iso")
    paths = {
        "home": root / "home",
        "config": root / "config",
        "state": root / "state",
        "data": root / "data",
        "cache": root / "cache",
        "tmux_tmpdir": root / "tmux",
    }
    for path in paths.values():
        path.mkdir(parents=True)
    for name in list(os.environ):
        if name in isolation.LEAKY_NAMES or name.startswith(isolation.LEAKY_PREFIXES):
            monkeypatch.delenv(name)
    settings = {
        "HOME": paths["home"],
        "XDG_CONFIG_HOME": paths["config"],
        "XDG_STATE_HOME": paths["state"],
        "XDG_DATA_HOME": paths["data"],
        "XDG_CACHE_HOME": paths["cache"],
        "TMUX_TMPDIR": paths["tmux_tmpdir"],
    }
    for name, path in settings.items():
        monkeypatch.setenv(name, str(path))
    problems = isolation.environment_problems(os.environ)
    assert not problems, "test environment is not isolated:\n" + "\n".join(problems)
    sub_env = {name: str(path) for name, path in settings.items()}
    sub_env["LANG"] = "C.UTF-8"
    yield Isolation(env=sub_env, **paths)
    problems = isolation.environment_problems(os.environ)
    assert not problems, "the test left a non-isolated environment:\n" + "\n".join(problems)


@pytest.fixture
def fakebin(tmp_path_factory: pytest.TempPathFactory, isolation_env: Isolation) -> Fakebin:
    """The fake claude-session, termctl, wtctl and tmux, with a log and a response script."""
    return Fakebin(tmp_path_factory.mktemp("fakebin") / "fb", isolation_env.env)


@pytest.fixture
def private_tmux(isolation_env: Isolation) -> Iterator[PrivateTmux]:
    """A real tmux server on a private socket (``-S <tmp>/sock -f /dev/null``), stopped after."""
    if REAL_TMUX is None:
        pytest.skip("tmux is not installed")
    server = PrivateTmux(REAL_TMUX, isolation_env.env)
    try:
        yield server
    finally:
        server.stop()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """`live` tests need the real world: never run them in CI, even with ``-m live``."""
    if not os.environ.get("CI"):
        return
    skip = pytest.mark.skip(reason="live tests never run in CI")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
