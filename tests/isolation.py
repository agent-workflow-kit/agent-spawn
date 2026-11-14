"""Keep tests away from the real home, XDG directories and tmux server.

``conftest.py`` points ``HOME`` and the ``XDG_*`` variables at per-test temporary directories
for every test and checks them with :func:`environment_problems`, which fails a test whose
environment names a directory of the real user (as seen when pytest started).
"""

from __future__ import annotations

import os
import pwd
from collections.abc import Mapping
from pathlib import Path

XDG_VARIABLES = {
    "XDG_CONFIG_HOME": ".config",
    "XDG_STATE_HOME": ".local/state",
    "XDG_DATA_HOME": ".local/share",
    "XDG_CACHE_HOME": ".cache",
}
LEAKY_PREFIXES = ("AGENT_SPAWN_", "TERMCTL_", "WTCTL_", "CSESSION_", "FAKEBIN_")
LEAKY_NAMES = ("TMUX", "TMUX_PANE", "XDG_RUNTIME_DIR")


#: Read once at import: tests may patch ``pwd`` or ``os.getuid`` later.
REAL_UID = os.getuid()
REAL_PASSWD_HOME = pwd.getpwuid(REAL_UID).pw_dir


def _real(path: str | Path) -> Path:
    return Path(os.path.realpath(path))


def _real_home_dirs(original: Mapping[str, str]) -> set[Path]:
    homes = {_real(REAL_PASSWD_HOME)}
    if original.get("HOME"):
        homes.add(_real(original["HOME"]))
    return homes


def _real_xdg_dirs(original: Mapping[str, str]) -> dict[str, set[Path]]:
    dirs: dict[str, set[Path]] = {}
    for name, default in XDG_VARIABLES.items():
        found = {_real(home / default) for home in _real_home_dirs(original)}
        if original.get(name):
            found.add(_real(original[name]))
        dirs[name] = found
    return dirs


def _real_tmux_parents(original: Mapping[str, str]) -> set[Path]:
    """Directories whose ``tmux-UID`` child is a real tmux socket directory."""
    tmpdirs = {_real("/tmp")}
    for entry in original.get("TMUX_TMPDIR", "").split(":"):
        if entry:
            tmpdirs.add(_real(entry))
    return tmpdirs


def _real_tmux_dirs(original: Mapping[str, str]) -> set[Path]:
    return {_real(d / f"tmux-{REAL_UID}") for d in _real_tmux_parents(original)}


def _inside(path: Path, parents: set[Path]) -> bool:
    return any(path == parent or parent in path.parents for parent in parents)


#: The environment pytest was started with; captured when ``conftest.py`` is imported.
ORIGINAL_ENV: dict[str, str] = dict(os.environ)


def environment_problems(
    env: Mapping[str, str], original: Mapping[str, str] | None = None
) -> list[str]:
    """Reasons ``env`` could reach the real user's home, XDG directories or tmux server."""
    original = ORIGINAL_ENV if original is None else original
    problems: list[str] = []
    home = env.get("HOME")
    if not home or not os.path.isabs(home):
        problems.append(f"HOME is not an absolute path: {home!r}")
    elif _real(home) in _real_home_dirs(original):
        problems.append(f"HOME is the real home directory: {home}")
    for name, real_dirs in _real_xdg_dirs(original).items():
        value = env.get(name)
        if not value or not os.path.isabs(value):
            problems.append(f"{name} is not an absolute path: {value!r}")
        elif _inside(_real(value), real_dirs):
            problems.append(f"{name} is (inside) the real directory: {value}")
    tmpdir = env.get("TMUX_TMPDIR")
    if not tmpdir or not os.path.isabs(tmpdir):
        problems.append(f"TMUX_TMPDIR is not an absolute path: {tmpdir!r}")
    else:
        real_tmpdir = _real(tmpdir)
        if (
            _inside(real_tmpdir, _real_tmux_dirs(original))
            or f"tmux-{REAL_UID}" in real_tmpdir.parts
        ):
            problems.append(f"TMUX_TMPDIR is the real tmux directory: {tmpdir}")
        elif real_tmpdir in _real_tmux_parents(original):
            # tmux looks for $TMUX_TMPDIR/tmux-UID/default: this is the parent of the real one
            problems.append(f"TMUX_TMPDIR holds the real tmux directory: {tmpdir}")
    for name in env:
        if name in LEAKY_NAMES or name.startswith(LEAKY_PREFIXES):
            problems.append(f"{name} is set in the test environment")
    return problems
