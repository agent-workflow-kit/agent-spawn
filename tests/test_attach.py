"""``attach``: termctl ``focus`` of the current run's window, against the fakes."""

from __future__ import annotations

import json
import os
import pty
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import attaching, term
from agent_spawn import gateway as gw
from agent_spawn.output import CliError
from agent_spawn.store import Store
from tests import schema_registry
from tests.conftest import Isolation
from tests.fake_gateway import FakeGateway
from tests.fakebin.harness import Fakebin
from tests.inspect_support import A1, A2, run_inprocess, seed_agent
from tests.launch_support import Env

ATTACH = "agent-spawn/cli/attach.json"
ERROR = "agent-spawn/cli/error.json"


class AttachEnv(Env):
    """An ``Env`` with two panes opened through the fake termctl (its ids and socket)."""

    t1: str
    t2: str
    sock: str

    def open_pane(self) -> tuple[str, str]:
        done = subprocess.run(
            [str(self.fakebin.bin_dir / "termctl"), "open", "--name", "w", "--label", f"agent={A1}",
             "--json", "--", "/bin/true"],
            capture_output=True, text=True, env=self.env(), stdin=subprocess.DEVNULL, check=True,
        )  # fmt: skip
        pane = json.loads(done.stdout)
        return pane["id"], pane["socket_path"]


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> AttachEnv:
    made = AttachEnv(fakebin, isolation_env)
    made.t1, made.sock = made.open_pane()
    made.t2, other = made.open_pane()
    assert other == made.sock
    return made


def focus_calls(fakebin: Fakebin) -> list[list[str]]:
    return [argv for argv in fakebin.argvs("termctl") if argv[0] == "focus"]


def out(done: Any, schema: str, code: int = 0) -> Any:
    assert done.returncode == code, (done.stdout, done.stderr)
    body = json.loads(done.stdout)
    schema_registry.validate(body, schema)
    return body


def live(env: AttachEnv, agent: str = A1, runs: Any = None, **fields: Any) -> None:
    runs = runs if runs is not None else [{"term_id": env.t1, "socket_path": env.sock}]
    seed_agent(env, agent, status="live", mode="interactive", run=len(runs), runs=runs, **fields)


def tmux_argv(sock: str, tmux: str = "/usr/bin/tmux", target: str = "=$1:@2.%3") -> list[str]:
    return [tmux, "-S", sock, "attach", "-E", "-t", target]


def focus_answer(env: AttachEnv, **fields: Any) -> dict[str, Any]:
    base = {
        "id": env.t1, "action": "attach", "client": None, "argv": tmux_argv(env.sock),
        "dry_run": False, "socket_path": env.sock, "session": "s", "window_id": "@2",
        "pane_id": "%3",
    }  # fmt: skip
    return {**base, **fields}


def test_switched_prints_the_golden_result_and_calls_focus_once(
    env: AttachEnv, fakebin: Fakebin
) -> None:
    live(env)
    body = out(env.run("attach", A1, "--json"), ATTACH)
    assert body == {
        "agent_id": A1, "action": "switched", "client": "/dev/pts/0", "argv": None,
        "term_id": env.t1, "socket_path": env.sock,
    }  # fmt: skip
    assert focus_calls(fakebin) == [["focus", env.t1, f"--socket={env.sock}", "--json"]]


def test_client_and_no_attach_are_passed_through(env: AttachEnv, fakebin: Fakebin) -> None:
    live(env)
    body = out(env.run("attach", A1, "--client", "pts/4", "--no-attach", "--json"), ATTACH)
    assert body["client"] == "/dev/pts/4"
    assert focus_calls(fakebin) == [
        ["focus", env.t1, f"--socket={env.sock}", "--client=pts/4", "--no-attach", "--json"]
    ]


def test_a_bad_client_is_a_usage_error_and_never_reaches_termctl(
    env: AttachEnv, fakebin: Fakebin
) -> None:
    live(env)
    done = env.run("attach", A1, "--client", "x;rm -rf", "--json")
    assert done.returncode == 2
    assert json.loads(done.stdout)["error"]["code"] == "usage"
    assert focus_calls(fakebin) == []


def test_attach_with_json_prints_the_argv_and_does_not_exec(
    env: AttachEnv, fakebin: Fakebin
) -> None:
    live(env)
    fakebin.script("termctl", "focus", stdout=focus_answer(env))
    body = out(env.run("attach", A1, "--json"), ATTACH)
    assert body["action"] == "attach" and body["client"] is None
    assert body["argv"] == tmux_argv(env.sock)


def test_attach_without_json_replaces_the_process_with_the_argv(
    env: AttachEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    live(env)
    record = tmp_path / "record"
    fake_tmux = tmp_path / "fake-tmux"
    fake_tmux.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {record}\nexit 7\n')
    fake_tmux.chmod(fake_tmux.stat().st_mode | stat.S_IXUSR)
    answer = focus_answer(env, argv=tmux_argv(env.sock, tmux=str(fake_tmux)))
    fakebin.script("termctl", "focus", stdout=answer)
    done = env.run("attach", A1)
    assert done.returncode == 7  # the exec'd program's own status: we were replaced
    assert record.read_text().split("\n")[:-1] == tmux_argv(env.sock)[1:]


def test_a_failed_exec_is_exit_12_with_a_message_not_a_traceback(
    env: AttachEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    live(env)
    answer = focus_answer(env, argv=tmux_argv(env.sock, tmux=str(tmp_path / "missing")))
    fakebin.script("termctl", "focus", stdout=answer)
    done = env.run("attach", A1)  # --json never execs, so a failed exec has no JSON form
    assert done.returncode == 12 and "Traceback" not in done.stderr
    assert done.stderr.startswith("agent-spawn: error: could not attach (")


def test_attach_without_json_hands_the_argv_to_exec_replace(
    env: AttachEnv, fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    live(env)
    fakebin.script("termctl", "focus", stdout=focus_answer(env))
    fake = FakeGateway()
    previous = gw.set_gateway(RoutedGateway(fake))
    try:
        code, stdout, _ = run_inprocess(env, monkeypatch, capsys, "attach", A1)
    finally:
        gw.set_gateway(previous)
    assert code == 0
    assert fake.exec_calls == [tmux_argv(env.sock)]
    assert stdout == f"{A1}: attach with {' '.join(tmux_argv(env.sock))}\n"


class RoutedGateway(gw.SystemGateway):
    """The real gateway, except that ``exec_replace`` is the fake's (it records and returns)."""

    def __init__(self, fake: FakeGateway) -> None:
        super().__init__()
        self._fake = fake

    def exec_replace(self, argv: Any) -> None:
        self._fake.exec_replace(argv)


@pytest.mark.parametrize(
    ("status", "hint"),
    [("parked", f"agent-spawn resume {A1}"), ("launching", None), ("tearing_down", None)],
)
def test_an_agent_that_is_not_live_is_exit_10_with_its_status(
    env: AttachEnv, fakebin: Fakebin, status: str, hint: str | None
) -> None:
    seed_agent(env, A1, status=status, mode="interactive", run=1,
               runs=[{"term_id": env.t1, "socket_path": env.sock}])  # fmt: skip
    body = out(env.run("attach", A1, "--json"), ERROR, 10)
    assert body["error"]["details"]["status"] == status
    assert body["error"]["details"].get("hint") == hint
    assert focus_calls(fakebin) == []


def test_torn_down_and_unknown_agents(env: AttachEnv, fakebin: Fakebin) -> None:
    seed_agent(env, A1, status="torn_down", torn_down_at=10.0)
    body = out(env.run("attach", A1, "--json"), ERROR, 10)
    assert body["error"]["details"]["status"] == "torn_down"
    body = out(env.run("attach", "ag-qqqqqqqq", "--json"), ERROR, 10)
    assert body["error"]["details"] == {"status": None}
    assert focus_calls(fakebin) == []


def test_an_ambiguous_prefix_is_exit_21_and_calls_nothing(env: AttachEnv, fakebin: Fakebin) -> None:
    live(env, "ag-aaaaaaaa")
    live(env, "ag-aaaabbbb", runs=[{"term_id": env.t2, "socket_path": env.sock}])
    body = out(env.run("attach", "ag-aaaa", "--json"), ERROR, 21)
    assert body["error"]["code"] == "ambiguous"
    assert focus_calls(fakebin) == []


@pytest.mark.parametrize(
    "make_runs",
    [
        lambda e: [{}],  # headless
        lambda e: [{"socket_path": e.sock}],  # an interrupted open: no pane recorded
        lambda e: [{"term_id": e.t1, "socket_path": e.sock}, {}],  # an older window is not used
    ],
)
def test_a_current_run_without_a_window_is_no_window(
    env: AttachEnv, fakebin: Fakebin, make_runs: Callable[[AttachEnv], list[dict[str, Any]]]
) -> None:
    live(env, runs=make_runs(env))
    body = out(env.run("attach", A1, "--json"), ERROR, 12)
    assert body["error"]["code"] == "no_window"
    assert "claude-session transcript" in body["error"]["details"]["hint"]
    assert focus_calls(fakebin) == []


def test_only_the_current_runs_window_is_focused(env: AttachEnv, fakebin: Fakebin) -> None:
    live(env, runs=[{"term_id": env.t1, "socket_path": env.sock},
                    {"term_id": env.t2, "socket_path": env.sock}])  # fmt: skip
    live(env, A2, runs=[{"term_id": env.t1, "socket_path": env.sock}])
    body = out(env.run("attach", A1, "--json"), ATTACH)
    assert body["term_id"] == env.t2
    assert focus_calls(fakebin) == [["focus", env.t2, f"--socket={env.sock}", "--json"]]


def test_a_stored_pane_id_that_is_not_an_id_never_reaches_termctl(
    env: AttachEnv, fakebin: Fakebin
) -> None:
    live(env, runs=[{"term_id": "--force", "socket_path": env.sock}])
    assert env.run("attach", A1, "--json").returncode != 0
    assert focus_calls(fakebin) == []


def switched(**fields: Any) -> Callable[[AttachEnv], dict[str, Any]]:
    def make(e: AttachEnv) -> dict[str, Any]:
        answer = focus_answer(e, **{"action": "switched", "client": "/dev/pts/1", **fields})
        if "argv" not in fields:
            del answer["argv"]
        return answer

    return make


BAD_ANSWERS: list[Callable[[AttachEnv], dict[str, Any]]] = [
    lambda e: focus_answer(e, id=e.t2),  # another pane
    lambda e: focus_answer(e, socket_path="/tmp/other.sock"),  # another server
    lambda e: focus_answer(e, argv=tmux_argv("/tmp/other.sock")),
    lambda e: focus_answer(e, argv=tmux_argv(e.sock, tmux="tmux")),  # not absolute
    lambda e: focus_answer(e, argv=tmux_argv(e.sock, target="=$1:@2.%3;rm")),
    lambda e: focus_answer(e, argv=[*tmux_argv(e.sock), "extra"]),
    lambda e: focus_answer(e, argv=["/bin/sh", "-c", "id"]),
    lambda e: focus_answer(e, argv=None),
    lambda e: focus_answer(e, client="/dev/pts/1"),  # attach never names a client
    lambda e: focus_answer(e, action="switched", client=None, argv=None),
    lambda e: focus_answer(e, action="switched", client="/dev/pts/1"),  # switched has no argv
    lambda e: focus_answer(e, action="other"),
    lambda e: {"id": e.t1},
    lambda e: focus_answer(e, dry_run=True),  # --dry-run was never passed
    lambda e: focus_answer(e, dry_run=None),
    lambda e: focus_answer(e, window_id="@9"),  # not the window in the target
    lambda e: focus_answer(e, pane_id="%9"),
    lambda e: focus_answer(e, window_id=2),
    lambda e: focus_answer(e, pane_id=None),
    lambda e: focus_answer(e, argv=tmux_argv(e.sock, target="=$1:@12.%3"), window_id="@2"),
    lambda e: focus_answer(e, argv=tmux_argv(e.sock, tmux="/usr/b\0in/tmux")),  # NUL
    switched(client="x;y"),  # a client name that is not one
    switched(dry_run=True),
    switched(argv=None),  # the pinned schema forbids argv on switched, even null
]


@pytest.mark.parametrize("make", BAD_ANSWERS)
def test_an_answer_that_is_not_the_asked_pane_or_command_is_refused(
    env: AttachEnv, fakebin: Fakebin, make: Callable[[AttachEnv], dict[str, Any]]
) -> None:
    live(env)
    fakebin.script("termctl", "focus", stdout=make(env))
    # with --json an accepted answer would exit 0, so 12 is the validator and not a failed exec
    body = out(env.run("attach", A1, "--json"), ERROR, 12)
    assert body["error"]["code"] == "failed"
    assert body["error"]["message"] == "termctl printed unusable output"


def test_a_refused_answer_is_never_exec_d(env: AttachEnv, fakebin: Fakebin, tmp_path: Path) -> None:
    live(env)
    record = tmp_path / "record"
    stub = tmp_path / "stub-tmux"
    stub.write_text(f"#!/bin/sh\necho ran > {record}\n")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    answer = focus_answer(env, argv=tmux_argv(env.sock, tmux=str(stub)), pane_id="%9")
    fakebin.script("termctl", "focus", stdout=answer)
    assert env.run("attach", A1).returncode == 12
    assert not record.exists()


def test_a_client_asked_for_with_an_attach_answer_is_refused(
    env: AttachEnv, fakebin: Fakebin
) -> None:
    live(env)
    fakebin.script("termctl", "focus", stdout=focus_answer(env))
    body = out(env.run("attach", A1, "--client", "pts/4", "--json"), ERROR, 12)
    assert body["error"]["message"] == "termctl printed unusable output"


def test_attach_is_refused_when_no_attach_was_asked(env: AttachEnv, fakebin: Fakebin) -> None:
    live(env)
    fakebin.script("termctl", "focus", stdout=focus_answer(env))
    assert env.run("attach", A1, "--no-attach", "--json").returncode == 12


def test_termctl_failures_pass_through(env: AttachEnv, fakebin: Fakebin) -> None:
    live(env)
    fakebin.script("termctl", "focus", times=1, exit=12,
                   error={"code": "no_client", "message": "x"})  # fmt: skip
    body = out(env.run("attach", A1, "--json"), ERROR, 12)
    assert body["error"]["tool"] == "termctl"
    fakebin.script("termctl", "focus", times=1, exit=10,
                   error={"code": "not_found", "message": "x"})  # fmt: skip
    assert out(env.run("attach", A1, "--json"), ERROR, 10)["error"]["tool"] == "termctl"


def test_switched_human_output_names_the_client(env: AttachEnv) -> None:
    live(env)
    done = env.run("attach", A1)
    assert done.returncode == 0 and "switched /dev/pts/0" in done.stdout


@pytest.mark.parametrize("setup", ["unknown", "parked", "headless"])
def test_a_bad_client_is_a_usage_error_before_the_agent_is_resolved(
    env: AttachEnv, fakebin: Fakebin, setup: str
) -> None:
    agent = A1
    if setup == "parked":
        seed_agent(env, A1, status="parked", mode="interactive", run=1,
                   runs=[{"term_id": env.t1, "socket_path": env.sock}])  # fmt: skip
    elif setup == "headless":
        live(env, runs=[{}])
    else:
        agent = "ag-qqqqqqqq"
    done = env.run("attach", agent, "--client", "x;y", "--json")
    assert json.loads(done.stdout)["error"]["code"] == "usage" and done.returncode == 2
    assert focus_calls(fakebin) == []


def test_a_hung_termctl_is_failed_after_the_timeout(
    env: AttachEnv, fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    live(env)
    fakebin.script("termctl", "focus", hang=True)
    monkeypatch.setattr(term, "FOCUS_TIMEOUT", 1.0)
    code, stdout, _ = run_inprocess(env, monkeypatch, capsys, "attach", A1, "--json")
    assert code == 12
    assert json.loads(stdout)["error"]["message"] == "termctl did not answer in time"


class ScriptedGateway(FakeGateway):
    def __init__(self, raises: BaseException) -> None:
        super().__init__()
        self.raises = raises

    def run_on_terminal(self, argv: Any, *, timeout: float | None = None) -> Any:
        raise self.raises


@pytest.mark.parametrize(
    ("raises", "code", "message"),
    [
        (FileNotFoundError(2, "no"), "tool_missing", "termctl is not installed or not executable"),
        (PermissionError(13, "no"), "tool_missing", "termctl is not installed or not executable"),
        (gw.OutputTooLarge("big"), "failed", "termctl printed too much output"),
        (OSError(5, "io"), "failed", "termctl could not run (io)"),
    ],
)
def test_focus_maps_runner_failures(raises: BaseException, code: str, message: str) -> None:
    backend = term.TermBackend(["/nonexistent/termctl"], gateway=ScriptedGateway(raises))
    with pytest.raises(CliError) as caught:
        backend.focus("tc-aaaaaaaa", "/tmp/s.sock")
    assert (caught.value.code, caught.value.message) == (code, message)


def test_focus_tool_missing_with_the_real_gateway() -> None:
    backend = term.TermBackend(["/nonexistent/termctl"], gateway=gw.SystemGateway())
    with pytest.raises(CliError) as caught:
        backend.focus("tc-aaaaaaaa", "/tmp/s.sock")
    assert caught.value.code == "tool_missing"


def test_a_vanished_agent_row_is_not_found(
    env: AttachEnv, fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    live(env)
    monkeypatch.setattr(Store, "get_agent", lambda self, agent_id: None)
    code, stdout, _ = run_inprocess(env, monkeypatch, capsys, "attach", A1, "--json")
    assert code == 10
    assert json.loads(stdout)["error"]["details"] == {"status": None}
    assert focus_calls(fakebin) == []


@pytest.mark.parametrize("run", [True, None, "1"])
def test_a_run_number_that_is_not_an_int_has_no_window(run: object) -> None:
    assert attaching._window(object(), {"agent_id": A1, "run": run}) is None  # pyright: ignore[reportPrivateUsage]


def test_attach_reaches_a_user_terminal_end_to_end(
    env: AttachEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    """The CLI run on a pty: termctl's own stdin and stderr are that terminal, and the exec'd
    program gets the argv termctl answered."""
    live(env)
    record = tmp_path / "record"
    tmux = tmp_path / "exec-stub"
    tmux.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {record}\nexit 7\n')
    bindir = tmp_path / "stubbin"
    bindir.mkdir()
    stub = bindir / "termctl"
    answer = json.dumps(focus_answer(env, argv=tmux_argv(env.sock, tmux=str(tmux))))
    stub.write_text(
        "#!/bin/sh\n"
        'if [ "$1" != focus ]; then exec ' + str(fakebin.bin_dir / "termctl") + ' "$@"; fi\n'
        "if [ -t 0 ] && [ -t 2 ]; then\n"
        f"  printf '%s\\n' '{answer}'\n"
        "else\n"
        '  printf \'{"error":{"code":"no_client","message":"no terminal"}}\\n\'\n'
        "  exit 12\n"
        "fi\n"
    )
    for script in (tmux, stub):
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    child_env = env.env()
    child_env["PATH"] = f"{bindir}:{child_env['PATH']}"
    master, slave = pty.openpty()
    try:
        done = subprocess.run(
            env.argv("attach", A1), stdin=slave, stderr=slave, stdout=subprocess.PIPE,
            env=child_env, timeout=60, check=False,
        )  # fmt: skip
    finally:
        os.close(slave)
        os.close(master)
    assert done.returncode == 7
    assert record.read_text().split("\n")[:-1] == tmux_argv(env.sock)[1:]
