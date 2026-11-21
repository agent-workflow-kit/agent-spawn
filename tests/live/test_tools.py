"""The tools themselves: versions and minimums, and the scenarios that are recorded, not run."""

from __future__ import annotations

import pytest

from .harness import MINIMUMS, World, check

SC = pytest.mark.live_scenario


@SC("tools-versions")
def test_versions_are_recorded_and_meet_the_minimums(world: World) -> None:
    """Records every tool's version (``LIVE_OBSERVATIONS``) so a run says what it ran against, and
    fails a tool older than the version the design was verified with. A version that cannot be
    read (an editable install reports 0.0.0) is recorded as unknown, never failed."""
    for name, tool in world.tools.items():
        if tool is None:
            world.observe(f"tool.{name}", "absent")
            continue
        world.observe(f"tool.{name}", tool.label)
        need = MINIMUMS.get(name)
        if need is None:
            continue
        ok = tool.at_least(need)
        if ok is False:
            check(
                False,
                f"{name} is at least {need}",
                "design.md Live verification (verified versions)",
                tool.label,
            )
        if ok is None:
            world.observe(f"tool.{name}.minimum", f"unknown version, needs {need}")
    cs = world.tools["claude-session"]
    assert cs is not None
    world.install_claude(stub_body="exit 0")
    world.claude_session_config()
    kinds = world.json_tool("claude-session", "kinds")
    check(
        kinds["claude_version_ok"] in (True, None) and kinds["min_claude_version"],
        "claude-session accepts the installed claude (kinds: claude_version_ok)",
        "design.md Uses / claude-session min_claude_version",
        {k: kinds[k] for k in ("claude_version", "claude_version_ok", "min_claude_version")},
    )
    world.observe("claude-session.min_claude_version", kinds["min_claude_version"])
    real = world.tools["claude"]
    assert real is not None
    check(
        real.at_least(kinds["min_claude_version"]) is not False,
        "the real claude is not older than claude-session's min_claude_version",
        "claude-session config min_claude_version",
        real.label,
    )
    check(world.tmux("-V").returncode == 0, "tmux runs", "design.md Uses")


@SC("placement-here-terminal")
def test_placement_here_needs_a_real_terminal() -> None:
    pytest.skip("needs a human at a real terminal (job control, resize, hangup)")


@SC("upgrade-live-supervisor")
def test_upgrade_under_a_live_supervisor() -> None:
    pytest.skip("needs two agent-spawn builds installed side by side")


@SC("real-crontab")
def test_real_crontab() -> None:
    pytest.skip("never touches the real crontab")


@SC("macos")
def test_macos() -> None:
    pytest.skip("macOS is out of scope for the Linux live suite")
