"""Every test runs with private HOME and XDG dirs and no AGENT_SPAWN_* variables."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_user_dirs(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    home: Path = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / ".local" / "state"))
    import os

    for name in [n for n in os.environ if n.startswith("AGENT_SPAWN_")]:
        monkeypatch.delenv(name)
