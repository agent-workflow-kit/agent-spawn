"""The fake runtime: invocation log, scripts, refusal outside the fixture, concurrency."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.fakebin.harness import FAKES, Fakebin, flag_values

HEADLESS = ("--mode", "headless", "--skill", "implement-issue")


def test_a_sample_test_asserts_a_fakes_logged_argv_exactly(fakebin: Fakebin) -> None:
    """The acceptance example: run through a fake, assert the exact argv it logged."""
    done = fakebin.run(
        "wtctl",
        "create",
        "--repo",
        "example.org/o/r",
        "--branch",
        "feat/x",
        "--holder",
        "ag-abcdefgh",
        "--label",
        "agent=ag-abcdefgh",
        "--actor",
        "auto:spawn",
        "--decision-id",
        "d-1",
        "--json",
    )
    assert done.returncode == 0
    fakebin.assert_argv(
        "wtctl",
        [
            "create",
            "--repo",
            "example.org/o/r",
            "--branch",
            "feat/x",
            "--holder",
            "ag-abcdefgh",
            "--label",
            "agent=ag-abcdefgh",
            "--actor",
            "auto:spawn",
            "--decision-id",
            "d-1",
            "--json",
        ],
    )
    call = fakebin.invocations("wtctl")[0]
    assert flag_values(call.argv, "--label") == ["agent=ag-abcdefgh"]
    assert fakebin.flag_values("wtctl", "--decision-id") == ["d-1"]
    assert fakebin.flag_values("wtctl", "--actor") == ["auto:spawn"]
    assert call.exit_code == 0 and call.cwd == os.getcwd()


def test_the_log_records_every_call_in_order_and_assertions_can_fail(fakebin: Fakebin) -> None:
    fakebin.run("termctl", "list", "--json")
    fakebin.run("wtctl", "list", "--json")
    fakebin.run("termctl", "exists", "tc-aaaaaaaa", "--json")
    assert [c.tool for c in fakebin.invocations()] == ["termctl", "wtctl", "termctl"]
    fakebin.assert_calls("termctl", [["list", "--json"], ["exists", "tc-aaaaaaaa", "--json"]])
    fakebin.assert_not_called("claude-session")
    try:
        fakebin.assert_argv("wtctl", ["list"])
    except AssertionError as failure:
        assert "got:" in str(failure)
    else:
        raise AssertionError("assert_argv accepted a different argv")


def test_the_log_carries_the_relevant_environment_but_not_arbitrary_variables(
    fakebin: Fakebin,
) -> None:
    fakebin.run("termctl", "list", "--json", env={"SECRET_TOKEN": "hunter2", "TERMCTL_SOCKET": "w"})
    env = fakebin.invocations("termctl")[0].env
    assert env["TERMCTL_SOCKET"] == "w" and "HOME" in env and "SECRET_TOKEN" not in env
    assert "hunter2" not in fakebin.log.read_text()


def test_a_fake_refuses_to_run_outside_the_fixture(tmp_path: Path, fakebin: Fakebin) -> None:
    env = {k: v for k, v in fakebin.env().items() if k != "FAKEBIN_LOG"}
    done = subprocess.run(
        [str(fakebin.bin_dir / "termctl"), "list"],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 70 and "FAKEBIN_LOG" in done.stderr


def test_the_fakes_shadow_the_real_tools_on_path(fakebin: Fakebin) -> None:
    for name in FAKES:
        found = subprocess.run(
            ["sh", "-c", f"command -v {name}"], env=fakebin.env(), capture_output=True, text=True
        )
        assert found.stdout.strip() == str(fakebin.bin_dir / name)


def test_rules_match_in_order_by_command_and_argv_and_are_limited_by_times(
    fakebin: Fakebin,
) -> None:
    fakebin.script(
        "termctl", "list", argv_contains=["--dead"], stdout={"socket_path": "/a", "panes": []}
    )
    fakebin.script("termctl", "list", times=1, merge={"socket_path": "/once"})
    assert (
        json.loads(fakebin.run("termctl", "list", "--dead", "--json").stdout)["socket_path"] == "/a"
    )
    assert json.loads(fakebin.run("termctl", "list", "--json").stdout)["socket_path"] == "/once"
    assert json.loads(fakebin.run("termctl", "list", "--json").stdout)["socket_path"] != "/once"


def test_a_scripted_error_prints_the_envelope_only_in_json_mode(fakebin: Fakebin) -> None:
    fakebin.script("wtctl", "list", error={"code": "internal", "message": "boom"}, exit=1)
    as_json = fakebin.run("wtctl", "list", "--json")
    assert as_json.returncode == 1 and json.loads(as_json.stdout)["error"]["message"] == "boom"
    plain = fakebin.run("wtctl", "list")
    assert plain.returncode == 1 and plain.stdout == "" and "boom" in plain.stderr


def test_stdout_text_scripts_and_exit_codes(fakebin: Fakebin) -> None:
    fakebin.script("termctl", "show", stdout="custom\n", exit=3)
    run = fakebin.run("termctl", "open", "--json", "--", "sleep", "1")
    assert run.returncode == 0
    shown = fakebin.run("termctl", "show", "tc-aaaaaaab")
    assert (shown.returncode, shown.stdout) == (3, "custom\n")


def test_concurrent_calls_all_land_in_the_log_and_ids_stay_unique(fakebin: Fakebin) -> None:
    def start(_: int) -> str:
        done = fakebin.run(
            "claude-session", "start", "--mode", "headless", "--skill", "s", "--json"
        )
        return json.loads(done.stdout)["csession_id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(start, range(16)))
    assert len(set(ids)) == 16
    calls = fakebin.invocations("claude-session")
    assert len(calls) == 16 and all(c.exit_code == 0 for c in calls)


def test_the_fakes_are_standard_library_only_and_python_is_the_running_interpreter_family(
    fakebin: Fakebin,
) -> None:
    for name in FAKES:
        text = (fakebin.bin_dir / name).resolve().read_text()
        assert text.startswith("#!/usr/bin/env python3")
    assert sys.version_info >= (3, 12)


# --- hung calls end (R16) --------------------------------------------------------------------


def test_a_hung_call_gives_up_after_the_bounded_time(fakebin: Fakebin) -> None:
    fakebin.seed_session()
    fakebin.script("claude-session", "wait", hang=True)
    done = fakebin.run(
        "claude-session",
        "wait",
        "cs-abcdefgh",
        "--json",
        env={"FAKEBIN_HANG_SECONDS": "0.5"},
        timeout=20,
    )
    assert done.returncode == 124 and "gave up after" in done.stderr


def test_a_hung_call_ends_when_its_parent_is_gone(fakebin: Fakebin) -> None:
    fakebin.seed_session()
    fakebin.script("claude-session", "wait", hang=True)
    code = (
        "import subprocess, sys\n"
        f"p = subprocess.Popen([{str(fakebin.bin_dir / 'claude-session')!r}, 'wait', "
        "'cs-abcdefgh', '--json'], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL, start_new_session=True)\n"
        "print(p.pid, flush=True)\n"
        "import time; time.sleep(1)\n"  # the fake has started and knows its parent
    )
    done = subprocess.run(  # the intermediate process exits, orphaning the hung fake
        [sys.executable, "-c", code], env=fakebin.env(), capture_output=True, text=True, timeout=20
    )
    pid = int(done.stdout)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        calls = [c for c in fakebin.invocations("claude-session") if c.pid == pid]
        if calls and calls[0].exit_code is not None:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("the orphaned fake did not exit")
    assert calls[0].exit_code == 124


# --- broken setups are one-line harness errors (R26, R27) ------------------------------------


def assert_harness_error(done: subprocess.CompletedProcess[str]) -> None:
    assert done.returncode == 70, (done.stdout, done.stderr)
    assert done.stderr.startswith("fakebin: ") and "Traceback" not in done.stderr
    assert len(done.stderr.strip().splitlines()) == 1


@pytest.mark.parametrize("content", ["{not json", "[]", '"text"'])
def test_a_corrupt_or_misshapen_state_file_is_a_harness_error(
    fakebin: Fakebin, content: str
) -> None:
    Path(str(fakebin.log) + ".sessions.json").write_text(content)
    assert_harness_error(fakebin.run("claude-session", "list", "--json"))


@pytest.mark.parametrize("content", ["{not json", "[]"])
def test_a_corrupt_counts_file_is_a_harness_error(fakebin: Fakebin, content: str) -> None:
    Path(str(fakebin.log) + ".counts.json").write_text(content)
    assert_harness_error(fakebin.run("claude-session", "start", "--json", *HEADLESS))


@pytest.mark.parametrize(
    "script",
    [
        "{not json",
        "[]",
        '{"rules": {}}',
        '{"rules": [1]}',
        '{"rules": [{"respond": []}]}',
        '{"rules": [{"respond": {"on_signal": {"SIGBOGUS": "ignore"}}}]}',
        '{"rules": [{"respond": {"on_signal": {"SIGTERM": 5}}}]}',
        '{"rules": [{"respond": {"sleep": "soon"}}]}',
        '{"rules": [{"respond": {"sleep": -1}}]}',
        '{"rules": [{"respond": {"exit": "two"}}]}',
        '{"rules": [{"respond": {"exit": 0, "error": {"code": "usage", "message": "m"}}}]}',
    ],
)
def test_a_bad_script_is_a_harness_error(fakebin: Fakebin, script: str) -> None:
    fakebin.script_file.write_text(script)
    assert_harness_error(fakebin.run("claude-session", "list", "--json"))


@pytest.mark.parametrize("log", ["relative.jsonl", "/nonexistent-dir-for-fakebin/log.jsonl"])
def test_an_unusable_log_path_is_a_harness_error(fakebin: Fakebin, log: str) -> None:
    done = fakebin.run("claude-session", "list", "--json", env={"FAKEBIN_LOG": log})
    assert_harness_error(done)


def test_a_scripted_error_without_exit_uses_the_codes_mapped_exit(fakebin: Fakebin) -> None:
    fakebin.script("wtctl", "list", times=1, error={"code": "refused", "message": "no"})
    fakebin.script("wtctl", "list", times=1, error={"code": "mystery", "message": "no"})
    fakebin.script("wtctl", "list", times=1, error={"code": "not_found", "message": "no"})
    codes = [fakebin.run("wtctl", "list", "--json").returncode for _ in range(3)]
    assert codes == [20, 1, 10]
    assert [c.exit_code for c in fakebin.invocations("wtctl")] == codes


def test_the_fakes_run_under_the_test_interpreter(fakebin: Fakebin) -> None:
    fakebin.run("termctl", "list", "--json")
    assert fakebin.python_versions() == [platform.python_version()]


@pytest.mark.parametrize(
    ("tool", "suffix", "args"),
    [("termctl", ".panes.json", ["list"]), ("wtctl", ".records.json", ["list"])],
)
def test_corrupt_termctl_and_wtctl_state_is_a_harness_error(
    fakebin: Fakebin, tool: str, suffix: str, args: list[str]
) -> None:
    Path(str(fakebin.log) + suffix).write_text("[1]")
    assert_harness_error(fakebin.run(tool, *args, "--json"))
