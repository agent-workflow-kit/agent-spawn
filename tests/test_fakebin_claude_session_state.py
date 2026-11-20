"""The fake claude-session's session state, signals, outcome files and argv acceptance.

Rows marked "real" were observed on the real claude-session 0.38.4 (argument parsing only; see
``tests/fakebin/fidelity_cases.py``); the rest follow ``docs/interface.md`` of claude-session.
"""

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
from tests.fakebin.harness import Fakebin
from tests.fakebin_support import error_of, run_json

HEADLESS = ("--mode", "headless", "--skill", "implement-issue")


def spawn(fakebin: Fakebin, *args: str, **kwargs: object) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [str(fakebin.bin_dir / "claude-session"), *args],
        stdout=kwargs.pop("stdout", subprocess.PIPE),  # type: ignore[arg-type]
        stderr=subprocess.PIPE,
        stdin=kwargs.pop("stdin", subprocess.DEVNULL),  # type: ignore[arg-type]
        text=True,
        env=fakebin.env(),
    )


def wait_for_call(fakebin: Fakebin, pid: int) -> None:
    """Wait until the hung fake is ready for signals (its handlers are installed)."""
    fakebin.wait_for_hang(pid)


def hanging_exec(
    fakebin: Fakebin, outcome_file: Path, mode: str = "headless"
) -> subprocess.Popen[str]:
    fakebin.script("claude-session", "exec", hang=True)
    proc = spawn(
        fakebin,
        "exec",
        "--mode",
        mode,
        "--skill",
        "x",
        "--json",
        "--outcome-file",
        str(outcome_file),
    )
    wait_for_call(fakebin, proc.pid)
    return proc


# --- signals (R1, R13, R18) ------------------------------------------------------------------


def send_together(proc: subprocess.Popen[str], numbers: tuple[signal.Signals, ...]) -> None:
    """Deliver ``numbers`` to ``proc`` (the PID this test launched) as one burst.

    The process is stopped first (and seen stopped), so every signal is pending when it is
    continued and the fake's handlers get all of them before it can finish. Sent one by one, a
    later signal can land after the fake began to exit, where Python has already put back the
    default action (the process then dies of that signal: -1 or -15, not 15).
    """
    proc.send_signal(signal.SIGSTOP)
    deadline = time.monotonic() + 10
    while True:
        info = os.waitid(os.P_PID, proc.pid, os.WSTOPPED | os.WNOHANG | os.WNOWAIT)
        if info is not None:
            break
        assert time.monotonic() < deadline, "the fake never stopped"
        time.sleep(0.005)
    for number in numbers:
        proc.send_signal(number)
    proc.send_signal(signal.SIGCONT)


def test_two_different_signals_in_a_row_still_finalize_once(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    outcome_file = tmp_path / "o.json"
    proc = hanging_exec(fakebin, outcome_file)
    send_together(proc, (signal.SIGTERM, signal.SIGHUP, signal.SIGTERM))
    out, err = proc.communicate(timeout=10)
    assert proc.returncode == 15 and "Traceback" not in err, err
    printed = json.loads(out)
    assert json.loads(outcome_file.read_text()) == printed and printed["outcome"] == "killed"
    last = fakebin.invocations("claude-session")[-1]
    # only the first signal counts (Python runs pending handlers by signal number)
    assert last.exit_code == 15 and len(last.signals) == 1


@pytest.mark.parametrize(
    ("number", "reason"),
    [
        (signal.SIGTERM, "forwarded_term"),
        (signal.SIGHUP, "forwarded_hup"),
        (signal.SIGINT, "forwarded_int"),
    ],
)
def test_a_signal_ends_a_headless_run_with_the_matching_end_reason(
    fakebin: Fakebin, tmp_path: Path, number: signal.Signals, reason: str
) -> None:
    proc = hanging_exec(fakebin, tmp_path / "o.json")
    proc.send_signal(number)
    out, _ = proc.communicate(timeout=10)
    printed = json.loads(out)
    sibling_schemas.validate(printed, "claude-session", "cli/exec.json")
    assert (proc.returncode, printed["end_reason"]) == (15, reason)


def test_sigint_is_ignored_by_an_interactive_run(fakebin: Fakebin, tmp_path: Path) -> None:
    fakebin.script("claude-session", "exec", hang=True)
    parent, child = pty.openpty()
    proc = subprocess.Popen(
        [str(fakebin.bin_dir / "claude-session"), "exec", "--skill", "x", "--json"],
        stdin=child,
        stdout=child,
        stderr=subprocess.PIPE,
        env=fakebin.env(),
    )
    try:
        wait_for_call(fakebin, proc.pid)
        proc.send_signal(signal.SIGINT)
        time.sleep(0.4)
        assert proc.poll() is None, "SIGINT must not end an interactive run"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == 15
    finally:
        if proc.poll() is None:
            proc.kill()  # the exact Popen child this test started
            proc.wait()
        if proc.stderr is not None:
            proc.stderr.close()
        os.close(parent)
        os.close(child)


def test_a_signal_after_stop_was_requested_ends_the_run_stopped(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    proc = hanging_exec(fakebin, tmp_path / "o.json")
    sid = json.loads(Path(str(fakebin.log) + ".sessions.json").read_text()).popitem()[0]
    run_json(fakebin, "claude-session", "stop", sid, schema="cli/stop.json")
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=10)
    assert json.loads(out)["end_reason"] == "stopped"


# --- outcome file failures (R5) --------------------------------------------------------------


@pytest.mark.parametrize("where", ["missing-parent", "directory", "empty"])
def test_a_failing_outcome_file_write_changes_neither_outcome_nor_exit_code(
    fakebin: Fakebin, tmp_path: Path, where: str
) -> None:
    target = {
        "missing-parent": str(tmp_path / "nope" / "o.json"),
        "directory": str(tmp_path),
        "empty": "",
    }[where]
    code, body = run_json(
        fakebin,
        "claude-session",
        "exec",
        *HEADLESS,
        "--outcome-file",
        target,
        schema="cli/exec.json",
    )
    assert code == 0 and body["outcome"] == "done"
    done = fakebin.run("claude-session", "exec", *HEADLESS, "--json", "--outcome-file", target)
    assert "warning: outcome file not written" in done.stderr and "Traceback" not in done.stderr
    assert len(fakebin.records("outcome_file_failed")) == 2
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


# --- state: known sessions (R10, R22), stop, resume, list (R6, R9) ---------------------------

ID_COMMANDS = [
    ["status"],
    ["stop"],
    ["wait"],
    ["forget"],
    ["resume", "--mode", "headless"],
    ["label", "add", "a=b"],
]


@pytest.mark.parametrize(
    "argv", ID_COMMANDS, ids=lambda a: a[0] + (a[1] if a[0] == "label" else "")
)
def test_a_never_created_full_id_exits_10(fakebin: Fakebin, argv: list[str]) -> None:
    head, *rest = argv
    if head == "label":
        full = ["label", "add", "cs-abcdefgh", "a=b"]
    else:
        full = [head, "cs-abcdefgh", *rest]
    code, error = error_of(fakebin, "claude-session", *full)
    assert (code, error["code"]) == (10, "not_found")


def test_adopting_or_resuming_a_never_created_id_exits_10(fakebin: Fakebin) -> None:
    assert error_of(fakebin, "claude-session", "exec", "--session-id", "cs-abcdefgh")[0] == 10
    assert error_of(fakebin, "claude-session", "start", "--resume", "cs-abcdefgh")[0] == 10


def test_label_add_on_an_unknown_id_creates_nothing(fakebin: Fakebin) -> None:
    error_of(fakebin, "claude-session", "label", "add", "cs-abcdefgh", "a=b")
    _, listed = run_json(fakebin, "claude-session", "list", schema="cli/list.json")
    assert listed["sessions"] == []
    assert error_of(fakebin, "claude-session", "status", "cs-abcdefgh")[0] == 10


def test_a_forgotten_id_is_unknown_and_a_seeded_one_is_known(fakebin: Fakebin) -> None:
    sid = fakebin.seed_session()
    assert run_json(fakebin, "claude-session", "status", sid, schema="cli/status.json")[0] == 0
    run_json(fakebin, "claude-session", "forget", sid, schema="cli/forget.json")
    assert error_of(fakebin, "claude-session", "status", sid)[0] == 10


@pytest.mark.parametrize(
    ("given", "code"),
    [("cs-abcdefghi", 10), ("cs-abc1efgh", 10), ("cs-abc", 2), ("abc", 2), ("CS-ABCDEFGH", 10)],
)
def test_malformed_ids_are_unknown_unless_shorter_than_four_characters(
    fakebin: Fakebin, given: str, code: int
) -> None:
    assert error_of(fakebin, "claude-session", "status", given)[0] == code


def test_stop_answers_by_the_run_state(fakebin: Fakebin) -> None:
    pending = fakebin.seed_session("cs-aaaaaaab", adopted=False, final=False)
    _, canceled = run_json(fakebin, "claude-session", "stop", pending, schema="cli/stop.json")
    assert canceled["canceled"] is True and canceled["signals"] == []
    _, again = run_json(fakebin, "claude-session", "stop", pending, schema="cli/stop.json")
    assert again["already_final"] is True and again["canceled"] is False
    live = fakebin.seed_session("cs-aaaaaaac", final=False, outcome=None)
    _, dry = run_json(fakebin, "claude-session", "stop", live, "--dry-run", schema="cli/stop.json")
    assert dry["ended"] is False
    _, ended = run_json(fakebin, "claude-session", "stop", live, schema="cli/stop.json")
    assert ended["ended"] is True and ended["outcome"] == "killed"
    final = fakebin.seed_session("cs-aaaaaaad")
    _, settled = run_json(fakebin, "claude-session", "stop", final, schema="cli/stop.json")
    assert settled["already_final"] is True and settled["outcome"] == "done"


def test_a_canceled_pending_run_cannot_be_adopted(fakebin: Fakebin) -> None:
    sid = fakebin.seed_session(adopted=False, final=False)
    run_json(fakebin, "claude-session", "stop", sid, schema="cli/stop.json")
    code, error = error_of(fakebin, "claude-session", "exec", "--session-id", sid)
    assert (code, error["code"]) == (20, "refused")


def test_resume_records_each_run_and_exec_continues_the_count(fakebin: Fakebin) -> None:
    sid = fakebin.seed_session()
    for expected in (2, 3):
        code, body = run_json(
            fakebin, "claude-session", "resume", sid, "--mode", "headless", schema="cli/resume.json"
        )
        assert (code, body["run"]) == (0, expected)
    _, started = run_json(
        fakebin, "claude-session", "start", "--resume", sid, schema="cli/start.json"
    )
    assert started["run"] == 4
    code, adopted = run_json(
        fakebin, "claude-session", *started["argv"][1:], schema="cli/exec.json"
    )
    assert (code, adopted["run"]) == (0, 4)


def test_resume_fork_allocates_a_new_session_with_forked_from(fakebin: Fakebin) -> None:
    sid = fakebin.seed_session(labels=["a=b"])
    code, body = run_json(
        fakebin,
        "claude-session",
        "resume",
        sid,
        "--fork",
        "--mode",
        "headless",
        schema="cli/resume.json",
    )
    assert code == 0 and body["csession_id"] != sid
    assert (body["forked_from"], body["run"]) == (sid, 1)
    _, status = run_json(
        fakebin, "claude-session", "status", body["csession_id"], schema="cli/status.json"
    )
    assert status["forked_from"] == sid
    _, original = run_json(fakebin, "claude-session", "status", sid, schema="cli/status.json")
    assert original["current_run"] == 1


def test_exec_makes_its_session_known(fakebin: Fakebin) -> None:
    _, body = run_json(fakebin, "claude-session", "exec", *HEADLESS, schema="cli/exec.json")
    code, status = run_json(
        fakebin, "claude-session", "status", body["csession_id"], schema="cli/status.json"
    )
    assert (code, status["state"]) == (0, "exited")


def test_list_applies_limit_state_kind_and_other_filters(fakebin: Fakebin) -> None:
    fakebin.seed_session("cs-aaaaaaab", kind="review", labels=["x=1"], mode="headless")
    fakebin.seed_session("cs-aaaaaaac", kind="implement", mode="interactive")
    fakebin.seed_session("cs-aaaaaaad", adopted=False, final=False, outcome=None)

    def ids(*args: str) -> list[str]:
        _, body = run_json(fakebin, "claude-session", "list", *args, schema="cli/list.json")
        return [item["csession_id"] for item in body["sessions"]]

    assert len(ids()) == 3
    assert ids("--limit", "2") == ["cs-aaaaaaab", "cs-aaaaaaac"]
    assert ids("--state", "starting") == ["cs-aaaaaaad"]
    assert ids("--state", "exited", "--state", "starting") == [
        "cs-aaaaaaab",
        "cs-aaaaaaac",
        "cs-aaaaaaad",
    ]
    assert ids("--kind", "review") == ["cs-aaaaaaab"]
    assert ids("--mode", "interactive") == ["cs-aaaaaaac"]
    assert ids("--pending") == ["cs-aaaaaaad"]
    assert ids("--live") == ["cs-aaaaaaad"]
    assert ids("--label", "x=1") == ["cs-aaaaaaab"]


def test_status_and_wait_refuse_a_run_that_does_not_exist(fakebin: Fakebin) -> None:
    sid = fakebin.seed_session()
    for command in ("status", "wait"):
        for run in ("0", "99"):
            assert error_of(fakebin, "claude-session", command, sid, "--run", run)[0] == 10
        assert error_of(fakebin, "claude-session", command, sid, "--run", "x")[0] == 2
    assert run_json(fakebin, "claude-session", "status", sid, "--run", "1")[0] == 0


# --- argv acceptance (R7, R8) ----------------------------------------------------------------

ACCEPTED = [
    pytest.param(["--env-allowlist-only", ""], id="empty-allowlist"),
    pytest.param(["--actor", "robot"], id="any-actor"),
    pytest.param(["--decision-id", "bad id"], id="any-decision-id"),
    pytest.param(["--permission-profile", "x/y"], id="file-form-profile"),
    pytest.param(["--max-turns", "+5"], id="max-turns-plus"),
    pytest.param(["--max-turns", " 5"], id="max-turns-space"),
    pytest.param(["--max-turns", "٥"], id="max-turns-arabic-digit"),
    pytest.param(["--auto-exit", "--no-auto-exit"], id="both-auto-exit-flags"),
    pytest.param(["--timeout", "1.5h"], id="fractional-timeout"),
    pytest.param(["--timeout", "none"], id="timeout-none"),
    pytest.param(["--setting-sources", ""], id="no-setting-sources"),
    pytest.param(["--setting-sources", "user,project"], id="setting-sources"),
    pytest.param(["--max-turns", "1_0"], id="max-turns-underscore"),
    pytest.param(["--hook", "pre-launch="], id="empty-hook-command"),
]


@pytest.mark.parametrize("extra", ACCEPTED)
def test_start_accepts_what_the_real_argv_parsing_accepts(
    fakebin: Fakebin, extra: list[str]
) -> None:
    code, _ = run_json(
        fakebin, "claude-session", "start", *HEADLESS, *extra, schema="cli/start.json"
    )
    assert code == 0


def test_help_without_a_command_exits_0_and_no_command_exits_2(fakebin: Fakebin) -> None:
    for flag in ("--help", "-h"):
        done = fakebin.run("claude-session", flag)
        assert done.returncode == 0 and done.stdout.startswith("usage:")
    bare = fakebin.run("claude-session")
    assert bare.returncode == 2
    assert fakebin.run("claude-session", "--json").returncode == 2


REFUSED = [
    pytest.param(["start", *HEADLESS, "--timeout", "36501d"], id="timeout-over-36500d"),
    pytest.param(["start", *HEADLESS, "--timeout", "1e3"], id="timeout-exponent"),
    pytest.param(["start", *HEADLESS, "--max-turns", "x"], id="max-turns-word"),
    pytest.param(["start", *HEADLESS, "--repo", "Bad Repo"], id="repo-grammar"),
    pytest.param(["start", *HEADLESS, "--repo", "host/../x"], id="repo-dotdot"),
    pytest.param(["start", *HEADLESS, "--env-allow", "A*"], id="env-allow-glob"),
    pytest.param(["start", *HEADLESS, "--env-allow", ""], id="env-allow-empty"),
    pytest.param(["start", *HEADLESS, "--env-allowlist-only", "A*"], id="allowlist-glob"),
    pytest.param(["start", *HEADLESS, "--setting-sources", "User"], id="sources-case"),
    pytest.param(["start", *HEADLESS, "--setting-sources", "user,"], id="sources-trailing-comma"),
    pytest.param(["start", *HEADLESS, "--permission-profile", "-x"], id="dash-profile"),
    pytest.param(["start", *HEADLESS, "--model", ""], id="empty-model"),
    pytest.param(["start", *HEADLESS, "--model=\t"], id="control-character-model"),
    pytest.param(["start", *HEADLESS, "--permission-profile=-x"], id="dash-profile-with-equals"),
    pytest.param(["start", *HEADLESS, "--permission-profile="], id="empty-profile"),
    pytest.param(["start", *HEADLESS, "--permission-profile=a b"], id="spaced-profile"),
    pytest.param(["start", "--resume", "cs-abcdefgh", "--reset", "kind"], id="reset-kind"),
    pytest.param(["forget", "--older-than", "abc"], id="forget-older-than-word"),
    pytest.param(["forget", "--older-than", "36501d"], id="forget-older-than-huge"),
    pytest.param(["forget", "--older-than", "1d", "--state", "bogus"], id="forget-state"),
    pytest.param(["forget", "--older-than", "1d", "--kind", "Bad"], id="forget-kind"),
    pytest.param(["list", "--limit", "0"], id="list-limit-0"),
    pytest.param(["list", "--limit", "x"], id="list-limit-word"),
    pytest.param(["list", "--state", "bogus"], id="list-state"),
    pytest.param(["list", "--kind", "Bad"], id="list-kind"),
    pytest.param(["list", "--kind", ""], id="list-kind-empty"),
    pytest.param(["list", "--report-status", "bogus"], id="list-report-status"),
    pytest.param(["list", "--mode", "x"], id="list-mode"),
    pytest.param(["list", "--since", "abc"], id="list-since"),
    pytest.param(["wait", "cs-abcdefgh", "--timeout", "36501d"], id="wait-timeout-huge"),
    pytest.param(["--json", "status", "cs-abcdefgh"], id="global-flag-before-command"),
    pytest.param(["--config", "x", "status", "cs-abcdefgh"], id="global-value-flag-before"),
    pytest.param(["label", "--json", "add", "cs-abcdefgh", "a=b"], id="label-flag-before-action"),
]


@pytest.mark.parametrize("argv", REFUSED)
def test_start_and_friends_refuse_what_the_real_argv_parsing_refuses(
    fakebin: Fakebin, argv: list[str]
) -> None:
    fakebin.seed_session()
    args = (
        [argv[0], "--json", *argv[1:]] if argv[0] in ("start", "forget", "list", "wait") else argv
    )
    done = fakebin.run("claude-session", *args)
    assert done.returncode == 2, (done.stdout, done.stderr)


# --- --implicit-done / --no-implicit-done (claude-session 0.43.0; grammar from its source, not
# yet in the live fidelity table: see FIDELITY.md) ---------------------------------------------


@pytest.mark.parametrize(
    ("extra", "effective"),
    [
        pytest.param(["--implicit-done"], True, id="implicit-done"),
        pytest.param(["--no-implicit-done"], False, id="no-implicit-done"),
        pytest.param(["--implicit-done", "--no-implicit-done"], False, id="both-last-wins"),
        pytest.param(["--no-implicit-done", "--implicit-done"], True, id="both-last-wins-reversed"),
    ],
)
def test_start_accepts_the_implicit_done_flags(
    fakebin: Fakebin, extra: list[str], effective: bool
) -> None:
    code, _ = run_json(
        fakebin, "claude-session", "start", *HEADLESS, *extra, schema="cli/start.json"
    )
    assert code == 0
    code, plan = run_json(
        fakebin, "claude-session", "start", *HEADLESS, *extra, "--dry-run",
        schema="cli/exec-dry-run.json",
    )  # fmt: skip
    assert code == 0
    assert plan["implicit_done"] is effective
    assert plan["profile"]["implicit_done"] is effective


@pytest.mark.parametrize("flag", ["--implicit-done", "--no-implicit-done"])
def test_start_refuses_a_value_for_the_implicit_done_flags(fakebin: Fakebin, flag: str) -> None:
    done = fakebin.run("claude-session", "start", "--json", *HEADLESS, f"{flag}=true")
    assert done.returncode == 2


@pytest.mark.parametrize("flag", ["--implicit-done", "--no-implicit-done"])
def test_reset_implicit_done_with_the_flag_is_a_usage_error(fakebin: Fakebin, flag: str) -> None:
    fakebin.seed_session()
    done = fakebin.run(
        "claude-session", "start", "--json", "--resume", "cs-abcdefgh",
        "--reset", "implicit-done", flag,
    )  # fmt: skip
    assert done.returncode == 2


def test_reset_implicit_done_alone_is_accepted(fakebin: Fakebin) -> None:
    fakebin.seed_session()
    done = fakebin.run(
        "claude-session", "start", "--json", "--resume", "cs-abcdefgh", "--reset", "implicit-done"
    )
    assert done.returncode == 0, (done.stdout, done.stderr)
