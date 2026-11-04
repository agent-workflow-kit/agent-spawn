from collections.abc import Iterator

import pytest

pytest_plugins = ["tests.procguard"]

_PRIVATE = (
    "HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_CACHE_HOME",
    "ZDOTDIR",
)


@pytest.fixture(autouse=True)
def private_environment(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Every test, and every program it starts, sees a private home and config, never the user's."""
    base = tmp_path_factory.mktemp("env")
    for name in _PRIVATE:
        path = base / name.lower()
        path.mkdir()
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("SHELL", "/bin/sh")
    yield
