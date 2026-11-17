"""``handoff.sweep_dead`` (#33): the guards on what it deletes, in process.

The command-level behavior (dead owner removed, live one kept) is in ``test_reconcile.py``; these
tests pin the refusals (symlinks, non-directories, another uid, a swapped directory), the clock
boundaries and the judging of the owner.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import handoff, procs
from agent_spawn.gateway import SystemGateway
from tests.test_teardown import dead, me

NOW = 2_000_000_000.0


class _Gateway(SystemGateway):
    """The system gateway with a fixed wall clock and an adjustable uid."""

    def __init__(self, uid_offset: int = 0) -> None:
        self.uid_offset = uid_offset

    def wall_time(self) -> float:
        return NOW

    def getuid(self) -> int:
        return os.getuid() + self.uid_offset


@pytest.fixture
def state(tmp_path: Path) -> Path:
    (tmp_path / "handoff").mkdir(mode=0o700)
    return tmp_path


def listener_dir(
    state: Path, name: str, owner: procs.ProcessIdentity | None, mtime: float | None = None
) -> Path:
    directory = state / "handoff" / name
    directory.mkdir(mode=0o700)
    (directory / "s").write_text("")
    if owner is not None:
        (directory / "owner").write_text(json.dumps(owner.to_dict()))
    if mtime is not None:
        os.utime(directory, (mtime, mtime))
    return directory


def sweep(state: Path, gateway: Any = None, **kwargs: Any) -> handoff.Sweep:
    return handoff.sweep_dead(state, gateway=gateway or _Gateway(), **kwargs)


def test_a_symlink_to_a_dead_listeners_directory_survives(state: Path, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere" / "aaaaaaaaaaaa"
    target.mkdir(parents=True)
    (target / "s").write_text("")
    (target / "owner").write_text(json.dumps(dead().to_dict()))
    os.symlink(target, state / "handoff" / "aaaaaaaaaaaa")
    result = sweep(state)
    assert result.removed == []
    assert (target / "s").exists() and (target / "owner").exists()
    assert os.path.islink(state / "handoff" / "aaaaaaaaaaaa")


def test_a_file_named_like_a_listener_directory_is_left_alone(state: Path) -> None:
    path = state / "handoff" / "aaaaaaaaaaaa"
    path.write_text("not a directory")
    assert sweep(state).removed == []
    assert path.read_text() == "not a directory"


def test_a_directory_of_another_uid_is_not_ours_to_delete(state: Path) -> None:
    directory = listener_dir(state, "aaaaaaaaaaaa", dead())
    result = sweep(state, _Gateway(uid_offset=1))
    assert directory.exists() and result.removed == []
    assert any("belongs to another user" in w for w in result.warnings)  # the root is not ours


def test_a_directory_swapped_between_the_lstat_and_the_open_is_not_removed(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = listener_dir(state, "aaaaaaaaaaaa", dead())
    real_open: Callable[..., int] = os.open
    swapped: list[Path] = []

    def swapping_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if path == "aaaaaaaaaaaa" and kwargs.get("dir_fd") is not None and not swapped:
            moved = state / "moved-away"
            os.rename(original, moved)
            replacement = listener_dir(state, "aaaaaaaaaaaa", dead())
            swapped.extend([moved, replacement])
        return real_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", swapping_open)
        result = sweep(state)
    moved, replacement = swapped
    assert result.removed == []
    assert (moved / "s").exists() and (replacement / "s").exists() and replacement.exists()


@pytest.mark.parametrize(("age", "removed"), [(3599.0, False), (3600.0, False), (3601.0, True)])
def test_a_directory_without_an_owner_goes_only_past_an_hour(
    state: Path, age: float, removed: bool
) -> None:
    directory = listener_dir(state, "aaaaaaaaaaaa", None, mtime=NOW - age)
    result = sweep(state)
    assert directory.exists() is not removed
    assert result.removed == (["aaaaaaaaaaaa"] if removed else [])


def test_a_live_process_of_another_boot_counts_as_dead(state: Path) -> None:
    mine = me()
    other_boot = procs.ProcessIdentity(mine.pid, mine.start_time, "another-boot")
    directory = listener_dir(state, "aaaaaaaaaaaa", other_boot)
    assert procs.is_alive(mine)  # the pid is live: only the boot id says "dead"
    assert sweep(state).removed == ["aaaaaaaaaaaa"] and not directory.exists()


def test_an_owner_that_cannot_be_judged_keeps_the_directory_with_a_warning(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(*args: Any, **kwargs: Any) -> bool:
        raise procs.ProcessIdentityError("cannot read the process entry")

    directory = listener_dir(state, "aaaaaaaaaaaa", me())
    monkeypatch.setattr(handoff.procs, "is_alive", unreadable)
    result = sweep(state)
    assert directory.exists() and result.removed == []
    assert result.warnings == ["handoff aaaaaaaaaaaa: its owner could not be judged: kept"]
