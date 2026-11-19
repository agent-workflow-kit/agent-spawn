"""The live suite's own helpers, tested with no real tool (they run in ``make verify``).

Covers the scenario registry and its agreement with the tests under ``tests/live/``, the
session counter and the ``claude`` wrapper (driven with a fake ``claude`` script), the private
environment and its leak check, credential handling and the login cache, the fake crontab, the
stub claude, process helpers and the gating (opt-in, selection, listing)."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.live import harness
from tests.live import scenarios as sc

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_DIR = REPO_ROOT / "tests" / "live"


# --- registry ------------------------------------------------------------------------------


def test_every_scenario_has_a_known_wave_and_tools() -> None:
    for scenario in sc.SCENARIOS.values():
        assert scenario.wave in sc.WAVES, scenario.name
        assert scenario.summary and scenario.covers, scenario.name
        for tool in scenario.tools:
            assert tool in harness.TOOL_ENV, (scenario.name, tool)
        assert set(scenario.optional) <= set(scenario.tools), scenario.name
        if scenario.billed:
            assert scenario.turns >= scenario.sessions or scenario.turns >= 1, scenario.name


def test_billed_scenarios_fit_the_default_session_cap() -> None:
    total = sc.expected_sessions()
    assert 0 < total <= harness.DEFAULT_MAX_SESSIONS


def test_scenarios_that_are_not_automated_say_why() -> None:
    skipped = [s for s in sc.SCENARIOS.values() if s.skip]
    assert {s.name for s in skipped} >= {"macos", "real-crontab", "placement-here-terminal"}
    for scenario in skipped:
        assert len(scenario.skip or "") > 20
        assert scenario.sessions == 0  # a skipped scenario never counts against the cap


def test_every_test_names_a_registered_scenario_and_every_scenario_has_a_test() -> None:
    used: dict[str, list[str]] = {}
    for path in sorted(LIVE_DIR.glob("test_*.py")):
        for name in re.findall(
            r"""(?:SC|live_scenario)\(\s*["']([a-z0-9-]+)["']\s*\)""", path.read_text()
        ):
            used.setdefault(name, []).append(path.name)
    assert not set(used) - set(sc.SCENARIOS), set(used) - set(sc.SCENARIOS)
    assert not set(sc.SCENARIOS) - set(used), set(sc.SCENARIOS) - set(used)


def test_selection_by_name_wave_and_class() -> None:
    assert sc.select(["headless-launch-wait"]) == ["headless-launch-wait"]
    wave = sc.select(["wave:billed-1"])
    assert wave and all(sc.SCENARIOS[n].wave == "billed-1" for n in wave)
    assert all(sc.SCENARIOS[n].billed for n in sc.select(["billed"]))
    assert not any(sc.SCENARIOS[n].billed for n in sc.select(["unbilled"]))
    assert sc.select(["all"]) == list(sc.SCENARIOS)
    assert sc.select(["tmux-real-server", "wave:tools"]).count("tmux-real-server") == 1


@pytest.mark.parametrize("token", ["nope", "wave:nope", "Billed", ""])
def test_an_unknown_selection_token_is_an_error_not_an_empty_selection(token: str) -> None:
    with pytest.raises(ValueError):
        sc.select([token])


def test_parse_selection_accepts_commas_and_spaces_and_unset() -> None:
    assert sc.parse_selection(None) is None
    assert sc.parse_selection("  ") is None
    assert sc.parse_selection("macos,tools-versions  tmux-real-server") == [
        "macos",
        "tools-versions",
        "tmux-real-server",
    ]


def test_the_listing_shows_sessions_turns_and_warns_over_the_cap() -> None:
    text = sc.format_listing(25)
    assert "headless-launch-wait" in text and "expected billed sessions" in text
    assert "SKIP:" in text
    small = sc.format_listing(2, ["resumable-after-sigterm"])
    assert "WARNING" in small
    assert "WARNING" not in sc.format_listing(25, ["tools-versions"])


def test_the_listing_command_prints_without_starting_anything() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "tests.live"],
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "LIVE_SCENARIOS": "wave:billed-1", "MAX_LIVE_SESSIONS": "7"},
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "interactive-window-close" in done.stdout and "MAX_LIVE_SESSIONS=7" in done.stdout
    assert "tmux-real-server" not in done.stdout
    bad = subprocess.run(
        [sys.executable, "-m", "tests.live"],
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "LIVE_SCENARIOS": "typo"},
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert bad.returncode == 2 and "unknown scenario" in bad.stderr


# --- environment variables ---------------------------------------------------------------------


def test_env_int_and_the_session_cap() -> None:
    assert harness.max_sessions({}) == 25
    assert harness.max_sessions({"MAX_LIVE_SESSIONS": "4"}) == 4
    assert harness.max_sessions({"MAX_LIVE_SESSIONS": ""}) == 25
    for bad in ("0", "-1", "x", "1.5", "٣"):
        with pytest.raises(ValueError):
            harness.max_sessions({"MAX_LIVE_SESSIONS": bad})


def test_opt_in_and_strict_modes() -> None:
    assert not harness.opted_in({}) and not harness.opted_in({"AS_LIVE": "yes"})
    assert harness.opted_in({"AS_LIVE": "1"})
    assert harness.strict({}) and not harness.strict({"AS_LIVE_STRICT": "0"})


# --- versions ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("claude-session 0.45.2", (0, 45, 2)),
        ("tmux 3.5a", (3, 5, 0)),
        ("2.1.328 (Claude Code)", (2, 1, 328)),
        ("git version 2.34.1", (2, 34, 1)),
        ("no version", None),
        ("", None),
    ],
)
def test_parse_version(text: str, expected: tuple[int, int, int] | None) -> None:
    assert harness.parse_version(text) == expected


def test_a_zero_version_is_unknown_not_old() -> None:
    assert harness.version_at_least((0, 0, 0), "0.43.0") is None
    assert harness.version_at_least(None, "0.43.0") is None
    assert harness.version_at_least((0, 43, 0), "0.43.0") is True
    assert harness.version_at_least((0, 42, 9), "0.43.0") is False
    assert harness.version_at_least((3, 5, 0), "3.2") is True


def test_tool_discovery_uses_the_override_then_path(tmp_path: Path) -> None:
    fake = tmp_path / "termctl"
    fake.write_text("#!/bin/sh\necho 'termctl 9.8.7'\n")
    fake.chmod(0o755)
    found = harness.discover_tools({"PATH": str(tmp_path)})
    assert found["termctl"] is not None and found["termctl"].version == (9, 8, 7)
    assert found["wtctl"] is None
    over = harness.discover_tools({"PATH": "/nonexistent", "LIVE_TERMCTL": str(fake)})
    assert over["termctl"] is not None and over["termctl"].path == str(fake)
    assert harness.discover_tools({"LIVE_TERMCTL": str(tmp_path / "nope")})["termctl"] is None


# --- credentials -------------------------------------------------------------------------------


def test_auth_prefers_the_api_key_then_the_token_and_hides_values(tmp_path: Path) -> None:
    both = {"ANTHROPIC_API_KEY": "sk-secret-1", "CLAUDE_CODE_OAUTH_TOKEN": "tok-secret-2"}
    auth = harness.resolve_auth(both, tmp_path)
    assert auth is not None and list(auth.env) == ["ANTHROPIC_API_KEY"]
    assert "secret" not in repr(auth) and "secret" not in str(auth)
    only = harness.resolve_auth({"CLAUDE_CODE_OAUTH_TOKEN": "tok-secret-2"}, tmp_path)
    assert only is not None and only.description == "CLAUDE_CODE_OAUTH_TOKEN"
    assert harness.resolve_auth({}, tmp_path) is None


def test_the_login_file_is_used_only_when_asked_for_and_never_a_symlink(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    creds = tmp_path / ".claude" / ".credentials.json"
    creds.write_text('{"x": "secret"}')
    assert harness.resolve_auth({}, tmp_path) is None  # not without the opt-in
    asked = harness.resolve_auth({"AS_LIVE_COPY_LOGIN": "1"}, tmp_path)
    assert asked is not None and asked.source == creds and "secret" not in repr(asked)
    creds.unlink()
    target = tmp_path / "elsewhere"
    target.write_text("x")
    creds.symlink_to(target)
    assert harness.resolve_auth({"AS_LIVE_COPY_LOGIN": "1"}, tmp_path) is None


def test_the_login_cache_keeps_private_modes_collects_refreshes_and_cleans_up(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real" / "credentials.json"
    source.parent.mkdir()
    source.write_text("original")
    original_mode = source.stat().st_mode
    cache = harness.new_login_cache(source, base=tmp_path)
    try:
        assert stat.S_IMODE(cache.directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(cache.path.stat().st_mode) == 0o600
        home = tmp_path / "home"
        cache.install(home)
        installed = home / ".claude" / ".credentials.json"
        assert (
            installed.read_text() == "original" and stat.S_IMODE(installed.stat().st_mode) == 0o600
        )
        installed.write_text("refreshed")  # claude refreshed the token in the private home
        cache.collect(home)
        assert cache.path.read_text() == "refreshed"
        assert (
            source.read_text() == "original" and source.stat().st_mode == original_mode
        )  # never written
        installed.unlink()
        installed.symlink_to(source)  # a symlink in the private home is never collected
        cache.collect(home)
        assert cache.path.read_text() == "refreshed"
    finally:
        cache.remove()
    assert not cache.directory.exists()


def test_claude_json_trusts_directories_and_approves_the_key_suffix() -> None:
    data = harness.claude_json(
        [Path("/a"), Path("/b")], api_key="sk-" + "x" * 30 + "ABCDEFGHIJKLMNOPQRST"
    )
    assert data["hasCompletedOnboarding"] is True
    assert data["projects"]["/a"]["hasTrustDialogAccepted"] is True
    assert data["customApiKeyResponses"]["approved"] == ["ABCDEFGHIJKLMNOPQRST"]
    assert "customApiKeyResponses" not in harness.claude_json([])


# --- the private environment ---------------------------------------------------------------------


def test_build_env_is_an_allowlist_and_private(tmp_path: Path) -> None:
    parent = {
        "HOME": "/home/real",
        "PATH": "/home/real/bin",
        "TMUX": "/tmp/tmux-1000/default,1,0",
        "SSH_AUTH_SOCK": "/run/agent",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "ANTHROPIC_API_KEY": "sk-secret",
        "GH_TOKEN": "g",
        "LANG": "de_DE.UTF-8",
        "HTTPS_PROXY": "http://proxy.invalid:3128",
    }
    root = tmp_path / "root"
    env = harness.build_env(parent, root=root)
    assert env["HOME"] == str(root / "home") and env["TMUX_TMPDIR"] == str(root / "tmuxdir")
    assert env["PATH"].split(os.pathsep)[0] == str(root / "bin") and "/home/real" not in env["PATH"]
    assert env["LANG"] == "de_DE.UTF-8" and env["HTTPS_PROXY"].startswith("http://proxy")
    for leaked in ("TMUX", "SSH_AUTH_SOCK", "XDG_RUNTIME_DIR", "ANTHROPIC_API_KEY", "GH_TOKEN"):
        assert leaked not in env
    assert not harness.environment_leaks(env, Path("/home/real"), root)


def test_only_the_billed_environment_carries_credentials(tmp_path: Path) -> None:
    auth = harness.Auth({"ANTHROPIC_API_KEY": "sk-secret"}, None, "ANTHROPIC_API_KEY")
    assert "ANTHROPIC_API_KEY" not in harness.build_env({}, root=tmp_path)
    assert harness.build_env({}, root=tmp_path, auth=auth)["ANTHROPIC_API_KEY"] == "sk-secret"


@pytest.mark.parametrize(
    "name",
    [
        "HOME",
        "PATH",
        "TMUX",
        "TMUX_TMPDIR",
        "XDG_STATE_HOME",
        "XDG_FOO",
        "CLAUDE_CONFIG_DIR",
        "SSH_AUTH_SOCK",
        "ANTHROPIC_API_KEY",
    ],
)
def test_a_test_may_not_override_the_private_environment(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        harness.build_env({}, root=tmp_path, extra={name: "x"})
    assert harness.build_env({}, root=tmp_path, extra={"MY_VAR": "1"})["MY_VAR"] == "1"


def test_the_leak_check_names_the_real_home_and_outside_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    env = harness.build_env({}, root=root)
    assert not harness.environment_leaks(env, tmp_path / "real", root)
    bad = {
        **env,
        "HOME": str(tmp_path / "real"),
        "TMUX_TMPDIR": "/tmp",
        "TMUX": "/tmp/x,1,0",
        "SSH_AUTH_SOCK": "/a",
    }
    problems = " ".join(harness.environment_leaks(bad, tmp_path / "real", root))
    assert (
        "HOME" in problems
        and "TMUX_TMPDIR" in problems
        and "TMUX names" in problems
        and "SSH_AUTH_SOCK" in problems
    )
    inside = {**env, "TMUX": f"{root}/tmuxdir/tmux-1/default,1,0"}
    assert not harness.environment_leaks(inside, tmp_path / "real", root)


# --- the claude wrapper and the session counter ------------------------------------------------


def make_wrapper(tmp_path: Path, maximum: int) -> tuple[Path, Path, Path]:
    real = tmp_path / "real-claude"
    log = tmp_path / "real-claude.args"
    real.write_text(f'#!/bin/sh\necho "$@" >> {log}\n')
    real.chmod(0o755)
    counter = tmp_path / "count"
    wrapper = tmp_path / "bin" / "claude"
    harness.write_claude_wrapper(wrapper, str(real), counter, maximum, "0.10")
    return wrapper, counter, log


def run_wrapper(wrapper: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(wrapper), *args],
        capture_output=True,
        text=True,
        timeout=30,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def test_the_wrapper_counts_launches_but_not_version_probes(tmp_path: Path) -> None:
    wrapper, counter, log = make_wrapper(tmp_path, 5)
    assert run_wrapper(wrapper, "--version").returncode == 0
    assert run_wrapper(wrapper, "-v").returncode == 0
    assert harness.counted_sessions(counter) == 0
    assert run_wrapper(wrapper, "--model", "haiku", "--", "hi there").returncode == 0
    assert harness.counted_sessions(counter) == 1
    assert "hi there" in log.read_text()


def test_the_wrapper_refuses_a_launch_over_the_cap_and_records_the_refusal(tmp_path: Path) -> None:
    wrapper, counter, log = make_wrapper(tmp_path, 2)
    for _ in range(2):
        assert run_wrapper(wrapper, "--model", "claude-haiku-4-5").returncode == 0
    third = run_wrapper(wrapper, "--model", "haiku")
    assert third.returncode == 97 and "MAX_LIVE_SESSIONS" in third.stderr
    assert log.read_text().count("\n") == 2  # the real claude never ran a third time
    assert harness.refusals(counter) and harness.refusals(counter)[0].startswith("over")


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--model", "opus"],
        ["--model", "sonnet", "-p"],
        ["-p", "haiku"],
        ["--model", "haiku", "--model", "opus"],  # the last --model wins in claude
        ["--model=haiku"],  # the inline form is not accepted
        ["--model", "haiku", "--model=opus"],
    ],
)
def test_the_wrapper_refuses_a_launch_without_a_haiku_model_before_counting(
    tmp_path: Path, args: list[str]
) -> None:
    wrapper, counter, log = make_wrapper(tmp_path, 5)
    done = run_wrapper(wrapper, *args)
    assert done.returncode == 98 and "haiku" in done.stderr
    assert harness.counted_sessions(counter) == 0 and not log.exists()
    assert harness.refusals(counter) == ["no-haiku"]


def test_the_wrapper_accepts_a_last_haiku_after_another_model(tmp_path: Path) -> None:
    wrapper, counter, _ = make_wrapper(tmp_path, 5)
    assert run_wrapper(wrapper, "--model", "opus", "--model", "haiku").returncode == 0
    assert harness.counted_sessions(counter) == 1


def test_the_wrapper_adds_a_budget_to_print_runs_only(tmp_path: Path) -> None:
    wrapper, _, log = make_wrapper(tmp_path, 9)
    run_wrapper(wrapper, "-p", "--model", "haiku", "--", "x")
    run_wrapper(wrapper, "--model", "haiku", "--", "y")
    run_wrapper(wrapper, "-p", "--max-budget-usd", "0.01", "--model", "haiku", "--", "z")
    lines = log.read_text().splitlines()
    assert lines[0].startswith("--max-budget-usd 0.10 -p")
    assert "--max-budget-usd" not in lines[1]
    assert lines[2].count("--max-budget-usd") == 1


def test_the_wrapper_quotes_hostile_paths(tmp_path: Path) -> None:
    real = tmp_path / "re al; echo pwned"
    real.write_text("#!/bin/sh\nexit 0\n")
    real.chmod(0o755)
    counter = tmp_path / "c o'unt"
    wrapper = tmp_path / "w"
    harness.write_claude_wrapper(wrapper, str(real), counter, 3)
    assert run_wrapper(wrapper, "--model", "haiku").returncode == 0
    assert harness.counted_sessions(counter) == 1


def test_the_session_budget_fails_before_a_test_that_cannot_finish(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    budget = harness.SessionBudget(counter, 3)
    budget.check_start("a", 3)
    counter.write_text("1 1\n2 2\n")
    assert budget.used == 2
    with pytest.raises(harness.LiveError, match="MAX_LIVE_SESSIONS"):
        budget.check_start("b", 2)
    budget.check_start("b", 1)


def test_the_session_budget_reports_overruns_and_refusals(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    budget = harness.SessionBudget(counter, 9)
    counter.write_text("1 1\n2 2\n3 3\n")
    assert budget.check_end("x", 3, 0) == []
    assert "registry says 1" in budget.check_end("x", 1, 0)[0]
    Path(f"{counter}.refused").write_text("over 10\n")
    assert "refused a launch" in budget.check_end("x", 9, 0)[0]
    # only refusals recorded after the test started count (the file is suite-wide)
    assert budget.check_end("x", 9, 0, refused_before=1) == []
    with Path(f"{counter}.refused").open("a") as handle:
        handle.write("no-haiku\n")
    assert "no-haiku" in budget.check_end("x", 9, 0, refused_before=1)[0]
    assert harness.refusals_since(counter, 1) == ["no-haiku"]


def test_a_scenario_with_several_tests_shares_its_sessions_between_them() -> None:
    sigterm = sc.SCENARIOS["resumable-after-sigterm"]
    assert sigterm.tests == 2 and sigterm.sessions_per_test * sigterm.tests >= sigterm.sessions
    assert sigterm.sessions_per_test == 2
    assert sc.SCENARIOS["headless-launch-wait"].sessions_per_test == 1


# --- fake crontab, stub claude, tripwire ---------------------------------------------------------


def run_crontab(
    path: Path, *args: str, text: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(path), *args], input=text, capture_output=True, text=True, timeout=30, check=False
    )


def test_the_fake_crontab_reads_writes_and_removes_only_its_own_table(tmp_path: Path) -> None:
    table = tmp_path / "table"
    cron = tmp_path / "crontab"
    harness.write_fake_crontab(cron, table)
    empty = run_crontab(cron, "-l")
    assert empty.returncode == 1 and "no crontab for" in empty.stderr
    assert run_crontab(cron, "-", text="* * * * * x\n").returncode == 0
    assert (
        run_crontab(cron, "-l").stdout == "* * * * * x\n" and table.read_text() == "* * * * * x\n"
    )
    assert run_crontab(cron, "-e").returncode == 2  # everything else is unsupported
    assert run_crontab(cron, "-r").returncode == 0 and not table.exists()


def test_the_tripwire_crontab_always_fails(tmp_path: Path) -> None:
    cron = tmp_path / "crontab"
    harness.write_tripwire_crontab(cron)
    done = run_crontab(cron, "-l")
    assert done.returncode == 70 and "tripwire" in done.stderr


def test_the_stub_claude_reports_a_version_records_its_pid_and_runs_its_body(
    tmp_path: Path,
) -> None:
    stub = tmp_path / "claude"
    pids = tmp_path / "pids"
    harness.write_stub_claude(stub, pids, "exit 7")
    assert run_crontab(stub, "--version").stdout.startswith("2.1.")
    assert not pids.exists()  # a version probe is not a run
    assert run_crontab(stub, "-p", "x").returncode == 7
    assert pids.read_text().strip().isdigit()


# --- processes -----------------------------------------------------------------------------------


def test_run_process_kills_exactly_its_own_group_on_a_timeout(tmp_path: Path) -> None:
    marker = tmp_path / "grandchild.pid"
    script = f"sleep 300 & echo $! > {marker}; wait"
    with pytest.raises(harness.LiveError, match="timed out"):
        harness.run_process(["sh", "-c", script], env={"PATH": "/usr/bin:/bin"}, timeout=1.5)
    pid = int(marker.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and harness.pid_alive(pid):
        time.sleep(0.05)
    assert not harness.pid_alive(pid)  # the group, grandchild included, is gone


def test_run_process_returns_output_and_passes_stdin(tmp_path: Path) -> None:
    done = harness.run_process(
        ["cat"], env={"PATH": "/usr/bin:/bin"}, timeout=10, input_text="hi\n"
    )
    assert (done.returncode, done.stdout) == (0, "hi\n")


def test_wait_for_polls_and_names_what_it_waited_for() -> None:
    seen = iter([0, 0, "yes"])
    assert harness.wait_for(lambda: next(seen), 5, "x", interval=0.01) == "yes"
    with pytest.raises(harness.LiveError, match="waiting for the moon"):
        harness.wait_for(lambda: False, 0.2, "the moon", interval=0.05)


def test_descendants_and_tokens_identify_processes() -> None:
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        token = harness.process_token(child.pid)
        assert token is not None and token == harness.process_token(child.pid)
        assert child.pid in harness.descendant_pids(os.getpid())
        assert harness.process_token(2**22 + 12345) is None
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait()


def test_check_raises_a_live_error_naming_assumption_and_document() -> None:
    harness.check(True, "fine", "doc")
    with pytest.raises(harness.LiveError) as info:
        harness.check(False, "the sky is blue", "design.md line 1", {"seen": 1})
    assert "the sky is blue" in str(info.value) and "design.md line 1" in str(info.value)
    assert isinstance(info.value, AssertionError)


# --- the world -----------------------------------------------------------------------------------


@pytest.fixture
def world() -> object:
    root = harness.new_root()
    tools = {
        "termctl": harness.Tool("termctl", "/bin/true", "termctl 1.2.3", (1, 2, 3)),
        "claude": None,
        "git": None,
        "tmux": None,
        "wtctl": None,
        "claude-session": None,
        "jobsctl": None,
    }
    w = harness.World(root, tools, {"LANG": "C.UTF-8"}, real_home=Path("/nonexistent"))
    yield w
    w.close()


def test_the_world_is_private_and_removed_at_close() -> None:
    root = harness.new_root()
    try:
        assert stat.S_IMODE(root.stat().st_mode) == 0o700 and str(root).startswith("/tmp/as-live-")
        assert stat.S_IMODE((root / "tmuxdir").stat().st_mode) == 0o700
        for name in ("home/.config", "home/.local/state", "tmp", "bin", "repos"):
            assert (root / name).is_dir()
    finally:
        shutil.rmtree(root)


def test_a_world_links_tools_writes_the_tripwire_and_builds_a_leak_free_env(
    world: harness.World,
) -> None:
    assert (world.bin / "termctl").is_symlink() and os.readlink(
        world.bin / "termctl"
    ) == "/bin/true"
    assert not (world.bin / "wtctl").exists()  # an absent tool gets no link
    assert run_crontab(world.bin / "crontab", "-l").returncode == 70
    env = world.env()
    assert env["PATH"].startswith(str(world.bin))
    assert "ANTHROPIC_API_KEY" not in env and "TMUX" not in env
    inner = world.env(tmux=world.default_socket)
    assert inner["TMUX"].startswith(str(world.root))
    with pytest.raises(ValueError):
        world.env({"HOME": "/home/real"})
    with pytest.raises(harness.LiveError, match="outside the private root"):
        world.env(tmux=Path("/tmp/elsewhere/default"))


def test_a_world_runs_children_in_the_private_environment(world: harness.World) -> None:
    done = world.run(["sh", "-c", "echo $HOME:$TMUX_TMPDIR:${TMUX:-unset}"])
    assert done.stdout.strip() == f"{world.home}:{world.tmuxdir}:unset"


def test_the_deadline_bounds_every_call(world: harness.World) -> None:
    world.deadline = time.monotonic() - 1
    with pytest.raises(harness.LiveTimeout):
        world.run(["true"])
    world.deadline = time.monotonic() + 3
    assert world.timeout(100) <= 3


def test_install_claude_stub_and_config_files(world: harness.World) -> None:
    world.install_claude(stub_body="exit 3")
    assert run_crontab(Path(world.claude_path), "-p").returncode == 3
    path = world.claude_session_config("lost_grace: 5s\n")
    assert (
        stat.S_IMODE(path.stat().st_mode) == 0o600
        and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    )
    text = path.read_text()
    assert f"claude_bin: {world.claude_path}" in text and "lost_grace: 5s" in text
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in text  # no credentials in a config file, ever


def test_the_oauth_token_variable_is_named_but_never_written(world: harness.World) -> None:
    world.auth = harness.Auth(
        {"CLAUDE_CODE_OAUTH_TOKEN": "tok-secret-value"}, None, "CLAUDE_CODE_OAUTH_TOKEN"
    )
    path = world.claude_session_config()
    assert "claude_auth_env: [CLAUDE_CODE_OAUTH_TOKEN]" in path.read_text()
    world.trust(Path("/some/dir"))
    for written in world.root.rglob("*"):
        if written.is_file() and not written.is_symlink():
            assert "tok-secret-value" not in written.read_text(errors="ignore"), written
    assert world.env(auth=True)["CLAUDE_CODE_OAUTH_TOKEN"] == "tok-secret-value"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in world.env()


def test_an_api_key_is_approved_by_suffix_in_the_private_claude_json(world: harness.World) -> None:
    world.auth = harness.Auth(
        {"ANTHROPIC_API_KEY": "sk-ant-" + "k" * 40}, None, "ANTHROPIC_API_KEY"
    )
    world.trust(Path("/trusted"))
    path = world.home / ".claude.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    data = json.loads(path.read_text())
    assert data["customApiKeyResponses"]["approved"] == ["k" * 20]
    assert "sk-ant" not in path.read_text()


def test_a_world_close_kills_only_tracked_processes_by_identity(world: harness.World) -> None:
    mine = subprocess.Popen(["sleep", "60"], start_new_session=True)
    other = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        world.track(mine.pid)
        world.pids.append(
            (other.pid, "not-its-token")
        )  # a recycled-PID lookalike is never signaled
        world.close()
        mine.wait(timeout=10)
        assert mine.returncode == -signal.SIGKILL
        assert other.poll() is None
    finally:
        other.kill()
        other.wait()
        mine.kill()
        mine.wait()


def fake_tmux(tmp_path: Path) -> tuple[harness.Tool, Path]:
    log = tmp_path / "tmux.calls"
    script = tmp_path / "fake-tmux"
    script.write_text(f'#!/bin/sh\necho "$@" >> {log}\n')
    script.chmod(0o755)
    return harness.Tool("tmux", str(script), "tmux 3.5a", (3, 5, 0)), log


def make_socket(path: Path) -> socket.socket:
    path.parent.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(path))
    return sock


def test_teardown_kills_every_private_server_even_after_the_deadline(tmp_path: Path) -> None:
    root = harness.new_root()
    tool, log = fake_tmux(tmp_path)
    w = harness.World(root, {"tmux": tool}, {"LANG": "C.UTF-8"}, deadline=time.monotonic() - 5)
    paths = [
        root / "tmuxdir" / f"tmux-{os.getuid()}" / "default",
        root / "tmuxdir" / "other" / "x.sock",
        root / "repos" / "deep" / "my-server",
    ]
    socks = [make_socket(p) for p in paths]
    try:
        with pytest.raises(harness.LiveTimeout):
            w.run(["true"])  # the deadline is over: ordinary calls refuse
        w.close()
        calls = log.read_text().splitlines()
        assert len(calls) == 3 and all(c.endswith("kill-server") for c in calls)
        assert {c.split()[3] for c in calls} == {str(p) for p in paths}
        assert not root.exists()
    finally:
        for sock in socks:
            sock.close()
        shutil.rmtree(root, ignore_errors=True)


def test_close_disarms_the_alarm_and_removes_the_root_when_collect_raises(
    tmp_path: Path,
) -> None:
    class Failing(harness.LoginCache):
        def collect(self, home: Path) -> None:
            raise OSError("disk trouble")

    source = tmp_path / "creds"
    source.write_text("x")
    cache = harness.new_login_cache(source, base=tmp_path)
    cache.__class__ = Failing
    root = harness.new_root()
    w = harness.World(root, {}, {"LANG": "C.UTF-8"}, login=cache)
    (root / "home" / ".claude" / ".credentials.json").write_text("token")
    signal.setitimer(signal.ITIMER_REAL, 600)
    try:
        with pytest.raises(OSError, match="disk trouble"):
            w.close()
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0  # disarmed first
        assert not root.exists()  # credentials and private home deleted regardless
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        cache.remove()
        shutil.rmtree(root, ignore_errors=True)


def test_close_kills_processes_running_in_the_world_by_home_or_cwd(world: harness.World) -> None:
    bare = {"PATH": "/usr/bin:/bin"}
    by_home = subprocess.Popen(
        ["sleep", "60"], env={**bare, "HOME": str(world.home)}, start_new_session=True
    )
    by_cwd = subprocess.Popen(
        ["sleep", "60"], cwd=world.root / "repos", env=bare, start_new_session=True
    )
    outside = subprocess.Popen(
        ["sleep", "60"], cwd="/", env={**bare, "HOME": "/nonexistent"}, start_new_session=True
    )
    try:
        found = {pid for pid, _ in harness.world_processes(world.root, world.home)}
        assert {by_home.pid, by_cwd.pid} <= found and outside.pid not in found
        world.close()
        by_home.wait(timeout=10)
        by_cwd.wait(timeout=10)
        assert by_home.returncode == -signal.SIGKILL and by_cwd.returncode == -signal.SIGKILL
        assert outside.poll() is None
    finally:
        for child in (by_home, by_cwd, outside):
            child.kill()
            child.wait()


def test_teardown_stops_the_noted_claude_sessions(tmp_path: Path) -> None:
    root = harness.new_root()
    log = tmp_path / "cs.calls"
    script = tmp_path / "cs"
    script.write_text(f'#!/bin/sh\necho "$@" >> {log}\n')
    script.chmod(0o755)
    tool = harness.Tool("claude-session", str(script), "claude-session 0.45.2", (0, 45, 2))
    w = harness.World(root, {"claude-session": tool}, {"LANG": "C.UTF-8"})
    w.note_session("cs-aaaaaaaa")
    w.note_session("cs-aaaaaaaa")
    w.close()
    assert log.read_text().splitlines() == ["stop cs-aaaaaaaa --json"]


def test_a_command_runs_in_the_private_repos_directory_by_default(world: harness.World) -> None:
    done = world.run(["pwd", "-P"])
    assert Path(done.stdout.strip()) == (world.root / "repos").resolve()
    assert Path.cwd() != world.root / "repos"
    other = world.run(["pwd", "-P"], cwd=world.home)
    assert Path(other.stdout.strip()) == world.home.resolve()


def test_termination_signals_raise_keyboard_interrupt_and_the_handlers_are_restored() -> None:
    before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP))
    restore = harness.install_termination_handlers()
    try:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            with pytest.raises(KeyboardInterrupt):
                os.kill(os.getpid(), sig)
                time.sleep(2)  # the handler runs before the sleep ends
    finally:
        restore()
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)) == before


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_make_repo_builds_a_wired_bare_origin_and_clone() -> None:
    root = harness.new_root()
    git = shutil.which("git")
    assert git is not None
    tools: dict[str, harness.Tool | None] = {name: None for name in harness.TOOL_ENV}
    tools["git"] = harness.Tool("git", git, "git version 2", (2, 0, 0))
    w = harness.World(root, tools, {}, real_home=Path("/nonexistent"))
    try:
        repo = w.make_repo("demo")
        assert repo.repo_id == "live.example/team/demo" and len(repo.sha) == 40
        assert (repo.clone / "README.md").exists() and (repo.origin / "HEAD").exists()
        assert w.git("rev-parse", "origin/main", cwd=repo.clone) == repo.sha
        config = w.wtctl_config({"demo": repo}).read_text()
        assert f"clone: {repo.clone}" in config and f"clone_url: {repo.origin}" in config
    finally:
        w.close()


# --- gating --------------------------------------------------------------------------------------


def pytest_run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    base = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CI", "AS_LIVE", "LIVE_SCENARIOS", "MAX_LIVE_SESSIONS")
    }
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


def test_asking_for_live_tests_without_the_opt_in_skips_every_one() -> None:
    done = pytest_run("tests/live", "-m", "live", "-q", "-rs", "-k", "not fidelity")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "AS_LIVE=1" in done.stdout
    assert " passed" not in done.stdout and " failed" not in done.stdout


def test_a_selection_deselects_the_other_scenarios_and_a_typo_is_a_usage_error() -> None:
    done = pytest_run(
        "tests/live",
        "-m",
        "live",
        "-q",
        "--collect-only",
        env={"AS_LIVE": "1", "LIVE_SCENARIOS": "termctl-focus"},
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "test_focus_without_a_client" in done.stdout and "test_wtctl" not in done.stdout
    typo = pytest_run(
        "tests/live",
        "-m",
        "live",
        "-q",
        "--collect-only",
        env={"AS_LIVE": "1", "LIVE_SCENARIOS": "typo"},
    )
    assert typo.returncode != 0 and "unknown scenario" in typo.stdout + typo.stderr


class _Item:
    def __init__(self, name: str | None, path: Path) -> None:
        self.path = path
        self.nodeid = f"{path.name}::t"
        self._name = name
        self.markers: list[str] = []

    def get_closest_marker(self, key: str) -> object | None:
        if key != "live_scenario" or self._name is None:
            return None
        return type("M", (), {"args": (self._name,)})()

    def add_marker(self, marker: object) -> None:
        self.markers.append(getattr(marker, "name", str(marker)))


def test_collection_rejects_a_live_test_without_a_registered_scenario() -> None:
    from tests.live import conftest as live_conftest

    config = type(
        "C", (), {"hook": type("H", (), {"pytest_deselected": staticmethod(lambda items: None)})}
    )()
    for name in (None, "not-a-scenario"):
        items = [_Item(name, LIVE_DIR / "test_x.py")]
        with pytest.raises(pytest.UsageError, match="must name a scenario"):
            live_conftest.pytest_collection_modifyitems(config, items)  # type: ignore[arg-type]
    outside = [_Item(None, REPO_ROOT / "tests" / "test_other.py")]
    live_conftest.pytest_collection_modifyitems(config, outside)  # type: ignore[arg-type]
    assert outside[0].markers == []  # tests outside tests/live are untouched


def test_the_make_targets_exist_and_verify_does_not_depend_on_live() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text()
    assert re.search(r"^live:\n\tAS_LIVE=1 uv run pytest -m live tests/live", makefile, re.M)
    assert re.search(r"^live-list:\n\tuv run python -m tests\.live", makefile, re.M)
    verify = next(line for line in makefile.splitlines() if line.startswith("verify:"))
    assert "live" not in verify
