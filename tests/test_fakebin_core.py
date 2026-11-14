"""The fake runtime: invocation log, scripts, refusal outside the fixture, concurrency."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tests.fakebin.harness import FAKES, Fakebin, flag_values


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
