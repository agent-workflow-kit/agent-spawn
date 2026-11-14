"""The fake claude-session: canned outputs validate against the pinned schemas, argv is checked."""

from __future__ import annotations

import json
import os
import pty
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tests import sibling_schemas
from tests.fakebin.fakelib.claude_session import OUTCOMES
from tests.fakebin.harness import Fakebin, flag_values
from tests.fakebin_support import error_of, run_json

HEADLESS = ("--mode", "headless", "--skill", "implement-issue")


def start(fakebin: Fakebin, *extra: str) -> dict[str, object]:
    code, body = run_json(
        fakebin, "claude-session", "start", *HEADLESS, *extra, schema="cli/start.json"
    )
    assert code == 0
    return body


def test_start_output_validates_and_names_the_adopting_argv(fakebin: Fakebin) -> None:
    body = start(fakebin, "--label", "agent=ag-abcdefgh")
    argv = body["argv"]
    assert isinstance(argv, list)
    assert os.path.isabs(argv[0]) and argv[1:] == [
        "exec",
        "--session-id",
        body["csession_id"],
        "--run",
        "1",
    ]
    assert body["run"] == 1 and body["forked_from"] is None


def test_adopting_the_printed_argv_writes_the_outcome_file_and_prints_the_outcome(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    outcome_file = tmp_path / "out.json"
    body = start(fakebin, "--outcome-file", str(outcome_file))
    argv = body["argv"]
    assert isinstance(argv, list)
    done = subprocess.run(
        [*argv, "--json"],
        capture_output=True,
        text=True,
        env=fakebin.env(),
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 0, done.stderr
    printed = json.loads(done.stdout)
    sibling_schemas.validate(printed, "claude-session", "cli/exec.json")
    assert json.loads(outcome_file.read_text()) == printed
    assert oct(outcome_file.stat().st_mode & 0o777) == "0o600"


def test_a_run_can_be_adopted_only_once(fakebin: Fakebin) -> None:
    body = start(fakebin)
    argv = body["argv"]
    assert isinstance(argv, list)
    first = subprocess.run(
        [*argv, "--json"], capture_output=True, env=fakebin.env(), stdin=subprocess.DEVNULL
    )
    assert first.returncode == 0
    second = subprocess.run(
        [*argv, "--json"],
        capture_output=True,
        text=True,
        env=fakebin.env(),
        stdin=subprocess.DEVNULL,
    )
    assert second.returncode == 20
    error = json.loads(second.stdout)["error"]
    assert error["code"] == "refused" and error["reason"] == "already_adopted"


def test_adopting_a_stale_run_is_refused_expired(fakebin: Fakebin) -> None:
    body = start(fakebin)
    sid = str(body["csession_id"])
    code, error = error_of(fakebin, "claude-session", "exec", "--session-id", sid, "--run", "7")
    assert (code, error["reason"]) == (20, "expired")


@pytest.mark.parametrize("outcome", sorted(OUTCOMES))
def test_every_scripted_outcome_validates_for_exec_wait_and_status(
    fakebin: Fakebin, outcome: str
) -> None:
    for command in ("exec", "wait", "status"):
        fakebin.script("claude-session", command, outcome=outcome)
    body = start(fakebin)
    sid = str(body["csession_id"])
    code, ran = run_json(
        fakebin, "claude-session", "exec", "--session-id", sid, schema="cli/exec.json"
    )
    assert code == OUTCOMES[outcome][0] == ran["exit_code"]
    code, waited = run_json(fakebin, "claude-session", "wait", sid, schema="cli/wait.json")
    assert code == OUTCOMES[outcome][0] and waited["outcome"] == outcome
    code, status = run_json(fakebin, "claude-session", "status", sid, schema="cli/status.json")
    assert code == 0 and status["run"]["outcome"] == outcome


def test_resume_dry_run_and_start_dry_run_validate(fakebin: Fakebin) -> None:
    body = start(fakebin)
    sid = str(body["csession_id"])
    code, plan = run_json(
        fakebin,
        "claude-session",
        "start",
        "--resume",
        sid,
        "--dry-run",
        schema="cli/exec-dry-run.json",
    )
    assert (code, plan["csession_id"], plan["run"], plan["fork"]) == (0, sid, 2, False)
    code, plan = run_json(
        fakebin,
        "claude-session",
        "start",
        "--resume",
        sid,
        "--fork",
        "--dry-run",
        schema="cli/exec-dry-run.json",
    )
    assert (plan["csession_id"], plan["forked_from"], plan["fork"]) == (None, sid, True)
    code, plan = run_json(
        fakebin, "claude-session", "start", *HEADLESS, "--dry-run", schema="cli/exec-dry-run.json"
    )
    assert plan["csession_id"] is None and plan["run"] == 1


def test_start_resume_adds_a_run_and_fork_allocates_a_new_id(fakebin: Fakebin) -> None:
    first = start(fakebin)
    sid = str(first["csession_id"])
    _, resumed = run_json(
        fakebin, "claude-session", "start", "--resume", sid, schema="cli/start.json"
    )
    assert (resumed["csession_id"], resumed["run"], resumed["forked_from"]) == (sid, 2, None)
    _, forked = run_json(
        fakebin, "claude-session", "start", "--resume", sid, "--fork", schema="cli/start.json"
    )
    assert forked["csession_id"] != sid and forked["forked_from"] == sid and forked["run"] == 1


def test_label_add_remove_stop_forget_list_reconcile_validate(fakebin: Fakebin) -> None:
    sid = str(start(fakebin, "--label", "agent=ag-abcdefgh")["csession_id"])
    code, added = run_json(
        fakebin,
        "claude-session",
        "label",
        "add",
        sid,
        "pr=1",
        "pr=1",
        "x=y",
        schema="cli/label-add.json",
    )
    assert added["added"] == ["pr=1", "x=y"] and added["labels"] == [
        "agent=ag-abcdefgh",
        "pr=1",
        "x=y",
    ]
    _, again = run_json(
        fakebin, "claude-session", "label", "add", sid, "pr=1", schema="cli/label-add.json"
    )
    assert again["added"] == []  # adding an existing pair is a no-op
    _, removed = run_json(
        fakebin,
        "claude-session",
        "label",
        "remove",
        sid,
        "pr=1",
        "nope=1",
        schema="cli/label-remove.json",
    )
    assert removed["removed"] == ["pr=1"]
    _, listed = run_json(
        fakebin,
        "claude-session",
        "list",
        "--label",
        "agent=ag-abcdefgh",
        "--label",
        "x=y",
        schema="cli/list.json",
    )
    assert [s["csession_id"] for s in listed["sessions"]] == [sid]
    _, none = run_json(
        fakebin, "claude-session", "list", "--label", "agent=other", schema="cli/list.json"
    )
    assert none["sessions"] == []
    _, stopped = run_json(fakebin, "claude-session", "stop", sid, schema="cli/stop.json")
    assert stopped["canceled"] and stopped["outcome"] == "not-started"  # still pending
    _, dry = run_json(fakebin, "claude-session", "stop", sid, "--dry-run", schema="cli/stop.json")
    assert dry["already_final"] is True and dry["signals"] == []
    _, forgotten = run_json(fakebin, "claude-session", "forget", sid, schema="cli/forget.json")
    assert [f["csession_id"] for f in forgotten["forgotten"]] == [sid]
    code, error = error_of(fakebin, "claude-session", "status", sid[:8])  # prefix of a forgotten id
    assert (code, error["code"]) == (10, "not_found")
    run_json(fakebin, "claude-session", "reconcile", schema="cli/reconcile.json")


def test_session_id_inputs_follow_the_documented_grammar(fakebin: Fakebin) -> None:
    sid = str(start(fakebin)["csession_id"])  # cs-aaaaaaab
    code, body = run_json(
        fakebin, "claude-session", "status", sid.upper()[3:], schema="cli/status.json"
    )
    assert (code, body["csession_id"]) == (0, sid)
    code, body = run_json(fakebin, "claude-session", "status", sid[:7], schema="cli/status.json")
    assert (code, body["csession_id"]) == (0, sid)
    assert error_of(fakebin, "claude-session", "status", "cs-aaa")[0] == 2  # fewer than 4
    start(fakebin)  # cs-aaaaaaac
    code, error = error_of(fakebin, "claude-session", "status", "aaaaaaa")
    assert (code, error["code"]) == (21, "ambiguous") and len(error["matches"]) == 2
    code, error = error_of(fakebin, "claude-session", "status", "mmmm")
    assert (code, error["code"]) == (10, "not_found")


def test_verify_secret_reads_the_secret_from_stdin(fakebin: Fakebin) -> None:
    fakebin.script("claude-session", "verify-secret", secret="s3cret", times=1)
    sid = "cs-abcdefgh"
    # verify-secret takes only its documented argv; --json is tolerated (seen on the real tool)
    first = fakebin.run("claude-session", "verify-secret", sid, input="s3cret\n")
    assert (first.returncode, json.loads(first.stdout)) == (0, {"match": True})
    second = fakebin.run("claude-session", "verify-secret", sid, input="s3cret\n")
    assert (second.returncode, json.loads(second.stdout)) == (11, {"match": False})
    sibling_schemas.validate(json.loads(second.stdout), "claude-session", "cli/verify-secret.json")
    assert fakebin.run("claude-session", "verify-secret", sid, "--json", input="x").returncode == 11
    assert [c.stdin for c in fakebin.invocations("claude-session")][:2] == ["s3cret\n", "s3cret\n"]
    # the secret is never in argv
    assert all("s3cret" not in " ".join(a) for a in fakebin.argvs("claude-session"))


def test_verify_secret_refuses_a_terminal_stdin_and_a_short_id_never_matches(
    fakebin: Fakebin,
) -> None:
    parent, child = pty.openpty()
    try:
        done = subprocess.run(
            [str(fakebin.bin_dir / "claude-session"), "verify-secret", "cs-abcdefgh"],
            stdin=child,
            capture_output=True,
            text=True,
            env=fakebin.env(),
        )
    finally:
        os.close(parent)
        os.close(child)
    assert done.returncode == 2
    # a short id is no match, not a usage error: the real tool gives no oracle
    assert fakebin.run("claude-session", "verify-secret", "cs-abcd", input="x").returncode == 11


USAGE_ERRORS = [
    pytest.param(["start", "--bogus"], id="unknown-flag"),
    pytest.param(["start", "--kind"], id="missing-value"),
    pytest.param(["start", "--skill", "x", "--mode", "headless", "extra"], id="stray-positional"),
    pytest.param(["start", "--mode", "headless"], id="headless-needs-skill-or-prompt"),
    pytest.param(["start", "--skill", "x", "--mode", "weird"], id="bad-mode"),
    pytest.param(["start", "--skill", "x", "--kind", "Bad Kind"], id="bad-kind"),
    pytest.param(["start", "--skill", "x", "--label", "Bad=1"], id="bad-label"),
    pytest.param(["start", "--skill", "x", "--timeout", "0"], id="timeout-zero"),
    pytest.param(["start", "--skill", "x", "--max-turns", "0"], id="max-turns-zero"),
    pytest.param(["start", "--skill", "x", "--model", "-x"], id="model-leading-dash"),
    pytest.param(["start", "--skill", "x", "--actor", "a" * 4097], id="actor-over-the-cap"),
    pytest.param(
        ["start", "--skill", "x", *["--decision-id", "d" * 100000] * 3],
        id="decision-ids-over-the-cap",
    ),
    pytest.param(["start", "--skill", "x", "--fork"], id="fork-without-resume"),
    pytest.param(["start", "--skill", "x", "--reset", "model"], id="reset-without-resume"),
    pytest.param(["start", "--skill", "x", "--allow-tainted"], id="allow-tainted-without-resume"),
    pytest.param(
        ["start", "--resume", "cs-abcdefgh", "--allow-tainted", "--actor", "auto:cron"],
        id="human-only-flag-with-auto-actor",
    ),
    pytest.param(
        ["start", "--resume", "cs-abcdefgh", "--allow-tainted"], id="human-only-flag-without-tty"
    ),
    pytest.param(
        [
            "start",
            "--resume",
            "cs-abcdefgh",
            "--fork",
            "--override-escalation",
            "--actor",
            "human:me",
        ],
        id="override-escalation-with-fork",
    ),
    pytest.param(
        ["start", "--resume", "cs-abcdefgh", "--reset", "model", "--model", "m"], id="reset-and-set"
    ),
    pytest.param(
        ["start", "--skill", "x", "--hook-timeout", "pre-launch=5s"], id="hook-timeout-alone"
    ),
    pytest.param(["start", "--skill", "x", "--hook", "bogus=true"], id="unknown-hook"),
    pytest.param(["start", "--skill", "x", "--input-file", "-"], id="input-file-dash"),
    pytest.param(
        ["start", "--skill", "x", "--prompt-file", "/nonexistent/prompt"], id="no-prompt-file"
    ),
    pytest.param(["start", "--skill", "x", "--verbose", "--quiet"], id="verbose-and-quiet"),
    pytest.param(["exec", "--run", "1"], id="exec-run-without-session-id"),
    pytest.param(
        ["exec", "--session-id", "cs-abcdefgh", "--kind", "review"], id="adopt-with-launch-flag"
    ),
    pytest.param(["exec", "--session-id", "cs-abcdefgh", "--run", "0"], id="run-not-positive"),
    pytest.param(["stop"], id="stop-without-id"),
    pytest.param(["wait", "cs-abcdefgh", "--timeout", "0"], id="wait-timeout-zero"),
    pytest.param(["label", "add", "cs-abcdefgh"], id="label-without-pair"),
    pytest.param(["label", "frob", "cs-abcdefgh", "a=b"], id="label-bad-subcommand"),
    pytest.param(["forget"], id="forget-nothing"),
    pytest.param(["forget", "cs-abcdefgh", "--older-than", "1d"], id="forget-id-and-bulk-flag"),
    pytest.param(["frobnicate"], id="unknown-command"),
]


@pytest.mark.parametrize("argv", USAGE_ERRORS)
def test_an_argv_the_real_tool_refuses_exits_2_with_the_usage_envelope(
    fakebin: Fakebin, argv: list[str]
) -> None:
    at = 2 if argv[0] == "label" else 1  # global flags follow the command name
    done = fakebin.run("claude-session", *argv[:at], "--json", *argv[at:])
    assert done.returncode == 2, done.stdout
    error = json.loads(done.stdout)
    sibling_schemas.validate(error, "claude-session", "cli/error.json")
    assert error["error"]["code"] == "usage"
    assert fakebin.invocations("claude-session")[-1].exit_code == 2


def test_start_requires_json(fakebin: Fakebin) -> None:
    done = fakebin.run("claude-session", "start", *HEADLESS)
    assert done.returncode == 2 and "requires --json" in done.stderr


def test_a_prompt_is_passed_by_file_never_as_a_positional(fakebin: Fakebin, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("do the thing")
    code, body = run_json(
        fakebin,
        "claude-session",
        "start",
        "--mode",
        "headless",
        "--prompt-file",
        str(prompt),
        schema="cli/start.json",
    )
    assert code == 0 and body["run"] == 1
    # a prompt text on argv is not part of the grammar (there is no `--`-separated prompt)
    refused = fakebin.run("claude-session", "start", "--json", "--mode", "headless", "--", "do it")
    assert refused.returncode == 2


def test_prompt_from_stdin_is_logged_and_flag_values_are_extractable(fakebin: Fakebin) -> None:
    done = fakebin.run(
        "claude-session",
        "start",
        "--mode",
        "headless",
        "--prompt-file",
        "-",
        "--decision-id",
        "d1",
        "--decision-id=d2",
        "--json",
        input="PROMPT\n",
    )
    assert done.returncode == 0
    call = fakebin.invocations("claude-session")[-1]
    assert call.stdin == "PROMPT\n" and "PROMPT" not in " ".join(call.argv)
    assert flag_values(call.argv, "--decision-id") == ["d1", "d2"]


def test_a_value_flag_takes_the_next_token_even_if_it_starts_with_a_dash(fakebin: Fakebin) -> None:
    # --label wants a value, and argparse takes "--json" for an option: a usage error
    done = fakebin.run(
        "claude-session", "start", "--mode", "headless", "--skill", "x", "--label", "--json"
    )
    assert done.returncode == 2


def test_signals_finalize_a_hanging_exec_as_killed_and_write_the_outcome_file(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    fakebin.script("claude-session", "exec", hang=True)
    outcome_file = tmp_path / "o.json"
    sid = str(start(fakebin, "--outcome-file", str(outcome_file))["csession_id"])
    proc = subprocess.Popen(
        [str(fakebin.bin_dir / "claude-session"), "exec", "--session-id", sid, "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=fakebin.env(),
        stdin=subprocess.DEVNULL,
    )
    _wait_for_invocation(fakebin, proc.pid)
    proc.send_signal(signal.SIGTERM)  # proc.pid is the PID this test launched
    out, _ = proc.communicate(timeout=10)
    assert proc.returncode == 15
    printed = json.loads(out)
    sibling_schemas.validate(printed, "claude-session", "cli/exec.json")
    assert printed["outcome"] == "killed" and json.loads(outcome_file.read_text()) == printed
    last = fakebin.invocations("claude-session")[-1]
    assert last.signals == [signal.SIGTERM] and last.exit_code == 15


def test_a_scripted_signal_response_can_ignore_sigterm(fakebin: Fakebin) -> None:
    fakebin.seed_session()
    fakebin.script("claude-session", "wait", hang=True, on_signal={"SIGTERM": "ignore"})
    proc = subprocess.Popen(
        [str(fakebin.bin_dir / "claude-session"), "wait", "cs-abcdefgh", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=fakebin.env(),
        stdin=subprocess.DEVNULL,
    )
    try:
        _wait_for_invocation(fakebin, proc.pid)
        proc.send_signal(signal.SIGTERM)
        time.sleep(0.3)
        assert proc.poll() is None, "SIGTERM should have been ignored"
    finally:
        proc.kill()  # the exact Popen child this test started
        proc.communicate()


def _wait_for_invocation(fakebin: Fakebin, pid: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if any(call.pid == pid for call in fakebin.invocations()):
            time.sleep(0.2)  # let the script's signal handlers install
            return
        time.sleep(0.05)
    raise AssertionError("the fake never logged its invocation")


def test_unimplemented_commands_say_so_distinctly(fakebin: Fakebin) -> None:
    done = fakebin.run("claude-session", "transcript", "cs-abcdefgh", "--json")
    # deliberately not a schema-valid code: a test must never mistake this for the real tool
    assert (done.returncode, json.loads(done.stdout)["error"]["code"]) == (1, "fake_unsupported")


def test_version_flag(fakebin: Fakebin) -> None:
    done = fakebin.run("claude-session", "--version")
    assert done.returncode == 0 and done.stdout == "claude-session 0.1.0\n"
