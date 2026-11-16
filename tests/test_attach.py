"""``attach``: termctl ``focus`` of the current run's window, against the fakes."""

from __future__ import annotations

import json
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.inspect_support import A1, A2, seed_agent
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


def tmux_argv(
    sock: str, tmux: str = "/usr/bin/tmux", target: str = "=$1:@2.%3"
) -> list[str]:
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


def test_a_failed_exec_is_exit_12_not_a_traceback(
    env: AttachEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    live(env)
    answer = focus_answer(env, argv=tmux_argv(env.sock, tmux=str(tmp_path / "missing")))
    fakebin.script("termctl", "focus", stdout=answer)
    done = env.run("attach", A1)
    assert done.returncode == 12 and "Traceback" not in done.stderr


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


def test_an_ambiguous_prefix_is_exit_21_and_calls_nothing(
    env: AttachEnv, fakebin: Fakebin
) -> None:
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
]


@pytest.mark.parametrize("make", BAD_ANSWERS)
def test_an_answer_that_is_not_the_asked_pane_or_command_is_refused(
    env: AttachEnv, fakebin: Fakebin, make: Callable[[AttachEnv], dict[str, Any]]
) -> None:
    live(env)
    fakebin.script("termctl", "focus", stdout=make(env))
    done = env.run("attach", A1)  # without --json: it would exec if it accepted the answer
    assert done.returncode == 12, (done.stdout, done.stderr)


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
