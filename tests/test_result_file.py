import json
import os
import stat
from pathlib import Path

import pytest

from agent_spawn import output
from agent_spawn.output import CliError


def _target(path: Path) -> output.ResultFileTarget:
    return output.resolve_result_file(str(path))


def test_resolve_stores_the_realpath_of_the_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    target = _target(tmp_path / "link" / "out.json")
    assert target.parent == str(real)
    assert target.name == "out.json"


def test_relative_path_is_resolved_against_the_cwd_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = output.resolve_result_file("out.json")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    assert target.parent == str(tmp_path.resolve())


@pytest.mark.parametrize("bad", ["", ".", "..", "dir/", "a/..", "x\0y", "/"])
def test_unusable_result_file_paths_are_usage_errors(bad: str, tmp_path: Path) -> None:
    with pytest.raises(CliError) as excinfo:
        output.resolve_result_file(bad)
    assert excinfo.value.exit_code == 2
    assert excinfo.value.code == "usage"


def test_write_creates_the_file_with_mode_0600_and_exact_content(tmp_path: Path) -> None:
    old_umask = os.umask(0)  # a permissive umask must not widen the mode
    try:
        warning = output.write_result_file(_target(tmp_path / "out.json"), '{"a":1}\n')
    finally:
        os.umask(old_umask)
    assert warning is None
    path = tmp_path / "out.json"
    assert path.read_text() == '{"a":1}\n'
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_write_replaces_an_existing_regular_file_atomically(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    path.write_text("old", encoding="utf-8")
    path.chmod(0o644)
    inode_before = path.stat().st_ino
    assert output.write_result_file(_target(path), "new") is None
    assert path.read_text() == "new"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.stat().st_ino != inode_before  # a rename, not an in-place rewrite


def test_write_leaves_no_temporary_files_behind(tmp_path: Path) -> None:
    output.write_result_file(_target(tmp_path / "out.json"), "x")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


def test_a_failed_write_leaves_the_previous_content_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "out.json"
    path.write_text("keep", encoding="utf-8")
    target = _target(path)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", boom)
    warning = output.write_result_file(target, "replacement")
    assert warning is not None
    assert "No space left" in warning
    assert path.read_text() == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


def test_symlink_swapped_into_a_parent_component_after_the_check_is_refused(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe"
    (safe / "sub").mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    target = _target(safe / "sub" / "out.json")
    # an attacker swaps a stored component for a link after the path was checked
    (safe / "sub").rmdir()
    (safe / "sub").symlink_to(victim)
    warning = output.write_result_file(target, "payload")
    assert warning is not None
    assert "path component 'sub' is a symbolic link" in warning
    assert list(victim.iterdir()) == []


def test_symlink_swapped_into_a_higher_component_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    victim = tmp_path / "victim"
    (victim / "b" / "c").mkdir(parents=True)
    target = _target(tmp_path / "a" / "b" / "c" / "out.json")
    (tmp_path / "a" / "b" / "c").rmdir()
    (tmp_path / "a" / "b").rmdir()
    (tmp_path / "a").rmdir()
    (tmp_path / "a").symlink_to(victim)
    warning = output.write_result_file(target, "payload")
    assert warning is not None
    assert not (victim / "b" / "c" / "out.json").exists()


def test_symlink_at_the_final_path_is_refused_and_its_target_untouched(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    link = tmp_path / "out.json"
    link.symlink_to(victim)
    warning = output.write_result_file(_target(link), "payload")
    assert warning is not None
    assert victim.read_text() == "precious"
    assert link.is_symlink()


def test_dangling_symlink_at_the_final_path_is_refused(tmp_path: Path) -> None:
    link = tmp_path / "out.json"
    link.symlink_to(tmp_path / "does-not-exist")
    assert output.write_result_file(_target(link), "payload") is not None
    assert not (tmp_path / "does-not-exist").exists()


def test_directory_at_the_final_path_is_a_warning(tmp_path: Path) -> None:
    (tmp_path / "out.json").mkdir()
    assert output.write_result_file(_target(tmp_path / "out.json"), "x") is not None


def test_missing_parent_directory_is_a_warning_not_an_exception(tmp_path: Path) -> None:
    warning = output.write_result_file(_target(tmp_path / "nope" / "out.json"), "x")
    assert warning is not None
    assert not (tmp_path / "nope").exists()


def test_pre_existing_temp_name_collision_does_not_follow_a_planted_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    monkeypatch.setattr(output, "_temp_name", lambda name: ".planted.tmp")
    (tmp_path / ".planted.tmp").symlink_to(victim)
    warning = output.write_result_file(_target(tmp_path / "out.json"), "payload")
    assert warning is not None  # O_EXCL refuses to reuse the planted name
    assert victim.read_text() == "precious"


def test_result_file_content_is_exactly_the_json_output(tmp_path: Path) -> None:
    payload = {"error": {"code": "usage", "message": "mé"}}
    text = output.render_json(payload) + "\n"
    output.write_result_file(_target(tmp_path / "r.json"), text)
    assert (tmp_path / "r.json").read_bytes() == text.encode("utf-8")
    assert json.loads((tmp_path / "r.json").read_text()) == payload
