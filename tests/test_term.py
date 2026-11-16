"""``term.py`` (the termctl calls) and the supervisor's window markers, against the fake termctl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import cli
from agent_spawn.ids import new_agent_id
from agent_spawn.output import CliError
from agent_spawn.store import Store
from agent_spawn.supervise import MARKER_BACKOFF, RunCtx, Signals, Supervisor
from agent_spawn.term import (
    CALL_TIMEOUT,
    MAX_AGENT_LABELS,
    OPEN_TIMEOUT,
    QUICK_TIMEOUT,
    TermBackend,
    window_labels,
)
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.session_backend_support import FakebinGateway

SOCKET = "/tmp/as-test-sock"


@pytest.fixture
def term(fakebin: Fakebin) -> TermBackend:
    return TermBackend(
        [str(fakebin.bin_dir / "termctl")],
        actor="auto:test",
        decision_ids=["d-1"],
        gateway=FakebinGateway(fakebin),
    )


def open_pane(term: TermBackend, name: str = "w", agent: str = "ag-aaaaaaaa") -> str:
    pane = term.open(
        name=name, labels=[f"agent={agent}"], placement=None, cwd="/tmp", argv=["/bin/true"]
    )
    assert pane.state == "running"
    return pane.term_id


# --- labels --------------------------------------------------------------------------------------


def test_window_labels_order_and_cap() -> None:
    stored = [f"k{i:02d}=v" for i in range(40)]
    labels, warnings = window_labels("implement", "ag-aaaaaaaa", stored)
    assert labels[:2] == ["kind=implement", "agent=ag-aaaaaaaa"]
    assert labels[2:] == stored[:MAX_AGENT_LABELS] and len(labels) == 32  # termctl's cap
    assert len(warnings) == 1 and "10 label(s)" in warnings[0]


def test_window_labels_without_a_usable_kind() -> None:
    assert window_labels(None, "ag-aaaaaaaa", [])[0] == ["agent=ag-aaaaaaaa"]
    labels, warnings = window_labels("bad kind;", "ag-aaaaaaaa", ["a=b"])
    assert labels == ["agent=ag-aaaaaaaa", "a=b"] and len(warnings) == 1


def test_window_labels_leave_out_a_pair_termctl_would_reject() -> None:
    labels, warnings = window_labels("k", "ag-aaaaaaaa", ["ok=1", "Bad Key=1", "x=a b\n"])
    assert labels == ["kind=k", "agent=ag-aaaaaaaa", "ok=1"] and len(warnings) == 2


# --- calls ---------------------------------------------------------------------------------------


def test_open_argv_uses_attached_values_and_no_shell(term: TermBackend, fakebin: Fakebin) -> None:
    pane = term.open(
        name="-x --json", labels=["agent=ag-aaaaaaaa"], placement="default", cwd="/tmp",
        argv=["/bin/true", "supervise", "--json"],
    )  # fmt: skip
    assert pane.term_id.startswith("tc-") and pane.pid is not None
    argv = fakebin.argvs("termctl")[-1]
    assert "--name=-x --json" in argv  # a value that looks like a flag stays a value
    assert argv[argv.index("--") + 1 :] == ["/bin/true", "supervise", "--json"]
    assert "--actor=auto:test" in argv and "--decision-id=d-1" in argv
    assert "--keep-on-exit=failed" in argv and "--json" in argv[: argv.index("--")]


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "nope"},
        {"state": "exploded"},
        {"pid": "7"},
        {"pid": -1},
        {"pid": True},
        {"socket_path": "relative/sock"},
        {"socket_path": None},
    ],
)
def test_open_output_that_is_not_a_pane_is_refused(
    term: TermBackend, fakebin: Fakebin, bad: dict[str, Any]
) -> None:
    fakebin.script("termctl", "open", merge=bad)
    with pytest.raises(CliError) as caught:
        open_pane(term)
    assert caught.value.code == "failed" and caught.value.tool == "termctl"


def test_open_exit_codes_map_through_the_envelope(term: TermBackend, fakebin: Fakebin) -> None:
    for exit_code, code in ((11, "vetoed"), (13, "escalated"), (12, "failed")):
        fakebin.clear_script("termctl", "open")
        fakebin.script("termctl", "open", exit=exit_code, error={"code": "x", "message": "m"})
        with pytest.raises(CliError) as caught:
            open_pane(term)
        assert (caught.value.code, caught.value.exit_code) == (code, exit_code)
        assert caught.value.lower_error == {"code": "x", "message": "m"}


def test_a_real_open_that_exits_2_is_a_placement_failure_not_internal(
    term: TermBackend, fakebin: Fakebin
) -> None:
    """The preflight accepted the request, so a later exit 2 (a directory that vanished, an
    argv over the preflight's size) is exit 12 `failed`, never the caller's usage error nor
    `internal`."""
    fakebin.script("termctl", "open", exit=2, error={"code": "usage", "message": "m"})
    with pytest.raises(CliError) as caught:
        open_pane(term)
    assert (caught.value.code, caught.value.exit_code) == ("failed", 12)
    assert caught.value.lower_code == "usage"


def test_the_preflight_maps_2_to_usage_and_everything_else_to_12(
    term: TermBackend, fakebin: Fakebin
) -> None:
    def run() -> CliError:
        with pytest.raises(CliError) as caught:
            term.preflight(name="w", labels=[], placement=None, cwd=None, argv=["/bin/true"])
        return caught.value

    fakebin.script("termctl", "open", exit=2, error={"code": "not_supported", "message": "m"})
    assert (run().code, run().exit_code, run().lower_code) == ("usage", 2, "not_supported")
    fakebin.clear_script("termctl", "open")
    for exit_code in (1, 10, 11, 12, 13, 20):
        fakebin.script("termctl", "open", exit=exit_code, error={"code": "x", "message": "m"})
        error = run()
        assert (error.code, error.exit_code, error.tool) == ("failed", 12, "termctl")
        assert "--placement here" in (error.details or {})["hint"]
        fakebin.clear_script("termctl", "open")


def test_a_missing_termctl_is_tool_missing(tmp_path: Path) -> None:
    backend = TermBackend([str(tmp_path / "nope")])
    with pytest.raises(CliError) as caught:
        backend.list_dead("ag-aaaaaaaa", SOCKET)
    assert caught.value.code == "tool_missing" and caught.value.tool == "termctl"


def test_list_dead_returns_only_valid_ids(term: TermBackend, fakebin: Fakebin) -> None:
    assert term.list_dead("ag-aaaaaaaa", SOCKET) == []
    fakebin.script("termctl", "list", merge={"panes": [{"id": "../x"}]})
    with pytest.raises(CliError):
        term.list_dead("ag-aaaaaaaa", SOCKET)
    fakebin.clear_script("termctl", "list")
    fakebin.script("termctl", "list", stdout={"socket_path": SOCKET, "panes": "x"})
    with pytest.raises(CliError):
        term.list_dead("ag-aaaaaaaa", SOCKET)


@pytest.mark.parametrize(
    "hostile", ["--force", "-h", "tc-AAAAAAAA", "tc-aaaaaaaa; x", "", "tc-aaaaaaaa\0"]
)
def test_a_pane_id_that_is_not_one_never_reaches_termctl(
    term: TermBackend, fakebin: Fakebin, hostile: str
) -> None:
    for call in (
        lambda: term.mark(hostile, SOCKET),
        lambda: term.unmark(hostile, SOCKET),
        lambda: term.close(hostile, SOCKET),
        lambda: term.label_add(hostile, SOCKET, ["a=b"]),
    ):
        with pytest.raises(CliError):
            call()
    fakebin.assert_not_called("termctl")


def test_a_nul_in_a_value_is_a_usage_error_before_the_call(
    term: TermBackend, fakebin: Fakebin
) -> None:
    with pytest.raises(CliError) as caught:
        term.mark("tc-aaaaaaaa", "/tmp/a\0b")
    assert caught.value.code == "usage"
    fakebin.assert_not_called("termctl")


class TimeoutSpy(FakebinGateway):
    """The fakebin gateway that records each call's timeout by termctl command."""

    def __init__(self, fakebin: Fakebin) -> None:
        super().__init__(fakebin)
        self.timeouts: dict[str, float | None] = {}

    def run(self, argv: Any, **kwargs: Any) -> Any:
        command = next(a for a in argv[1:] if not a.startswith("-"))
        self.timeouts[command] = kwargs.get("timeout")
        return super().run(argv, **kwargs)


def test_only_open_and_close_may_take_long_the_rest_are_quick(fakebin: Fakebin) -> None:
    spy = TimeoutSpy(fakebin)
    backend = TermBackend([str(fakebin.bin_dir / "termctl")], gateway=spy)
    pane = backend.open(
        name="w", labels=["agent=ag-aaaaaaaa"], placement=None, cwd="/tmp", argv=["/bin/true"]
    )
    backend.mark(pane.term_id, pane.socket_path)
    backend.unmark(pane.term_id, pane.socket_path)
    backend.label_add(pane.term_id, pane.socket_path, ["a=1"])
    backend.label_remove(pane.term_id, pane.socket_path, ["a=1"])
    backend.list_dead("ag-aaaaaaaa", pane.socket_path)
    backend.find_panes("ag-aaaaaaaa", pane.socket_path)
    assert spy.timeouts["open"] == OPEN_TIMEOUT
    for command in ("mark", "unmark", "label", "list"):
        assert spy.timeouts[command] == QUICK_TIMEOUT, command
    backend.list_dead("ag-aaaaaaaa", pane.socket_path)
    with pytest.raises(CliError):  # the pane is running: refused, but the call was made
        backend.close(pane.term_id, pane.socket_path)  # may run hooks: the long timeout
    assert spy.timeouts["close"] == CALL_TIMEOUT
    assert QUICK_TIMEOUT <= 5.0


def test_find_panes_lists_by_label_on_the_given_server(term: TermBackend, fakebin: Fakebin) -> None:
    mine = open_pane(term, agent="ag-aaaaaaaa")
    open_pane(term, agent="ag-bbbbbbbb")
    found = term.find_panes("ag-aaaaaaaa", None)
    assert [term_id for term_id, _ in found] == [mine]
    assert term.find_panes("ag-aaaaaaaa", found[0][1]) == found
    argv = fakebin.argvs("termctl")[-1]
    assert "--label=agent=ag-aaaaaaaa" in argv and "--dead" not in argv


# --- the supervisor's markers --------------------------------------------------------------------


class MarkerRig:
    """A window-hosted supervisor whose current run (2) has a window and an earlier run (1) too."""

    def __init__(self, fakebin: Fakebin, iso: Isolation, tmp_path: Path) -> None:
        self.fakebin = fakebin
        self.state = tmp_path / "state"
        self.backend = TermBackend(
            [str(fakebin.bin_dir / "termctl")], gateway=FakebinGateway(fakebin)
        )
        self.first = open_pane(self.backend, "first")
        self.second = open_pane(self.backend, "second")
        with Store.open(self.state) as store:
            self.agent_id = new_agent_id(store)
            store.insert_agent(
                self.agent_id,
                {"status": "live", "name": "n", "mode": "interactive", "run": 2},
            )
            with store.lease(f"agent:{self.agent_id}") as lease:
                store.put_run(
                    lease, self.agent_id, 1, {"term_id": self.first, "socket_path": self.sock()}
                )
                store.put_run(
                    lease, self.agent_id, 2, {"term_id": self.second, "socket_path": self.sock()}
                )
        config = tmp_path / "config.yaml"
        inv = cli.parse_invocation(
            [
                "supervise", self.agent_id, "--handoff", "/x", "--handoff-timeout", "1",
                "--state-dir", str(self.state), "--config", str(config),
            ]
        )  # fmt: skip
        self.supervisor = Supervisor(inv, Signals(interactive=True))
        self.supervisor.store = Store.open(self.state, create=False)
        self.supervisor.term = self.backend

    def sock(self) -> str:
        done = self.fakebin.run("termctl", "show", self.first, "--json")
        return json.loads(done.stdout)["socket_path"]

    def relay(self, run: int, state: str | None, activity: str | None) -> None:
        assert self.supervisor.store is not None
        with self.supervisor.store.lease(f"agent:{self.agent_id}") as lease:
            self.supervisor.store.put_run(
                lease, self.agent_id, run, {"session_state": state, "activity": activity}
            )

    def marker(self, term_id: str) -> str | None:
        return json.loads(self.fakebin.run("termctl", "show", term_id, "--json").stdout)["marker"]

    def calls(self, command: str) -> int:
        return len([a for a in self.fakebin.argvs("termctl") if a[0] == command])

    def close(self) -> None:
        assert self.supervisor.store is not None
        self.supervisor.store.close()


@pytest.fixture
def rig(fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path) -> Any:
    made = MarkerRig(fakebin, isolation_env, tmp_path)
    try:
        yield made
    finally:
        made.close()


def test_needs_input_and_idle_mark_and_working_clears(rig: MarkerRig) -> None:
    ctx = RunCtx(2, "/x")
    rig.relay(2, "needs-input", None)
    rig.supervisor.apply_marker(ctx)
    assert rig.marker(rig.second) == "attention" and rig.marker(rig.first) is None
    rig.relay(2, "working", "working")
    rig.supervisor.apply_marker(ctx)
    assert rig.marker(rig.second) is None
    rig.relay(2, "idle", "idle")
    rig.supervisor.apply_marker(ctx)
    assert rig.marker(rig.second) == "attention"


class Clock:
    """A settable clock for the supervisor's marker backoff (no sleeping in a test)."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_repeats_are_not_sent_and_a_failure_is_retried_after_the_backoff(rig: MarkerRig) -> None:
    ctx = RunCtx(2, "/x")
    clock = Clock()
    rig.supervisor.clock = clock
    rig.relay(2, "needs-input", None)
    rig.fakebin.script(
        "termctl", "mark", times=1, exit=12, error={"code": "locked", "message": "m"}
    )
    rig.supervisor.apply_marker(ctx)
    assert rig.marker(rig.second) is None  # a warning, not a crash
    rig.supervisor.apply_marker(ctx)
    assert rig.calls("mark") == 1  # backing off: no call on the next look
    clock.now += MARKER_BACKOFF + 1
    rig.supervisor.apply_marker(ctx)
    assert rig.marker(rig.second) == "attention"
    before = rig.calls("mark")
    rig.supervisor.apply_marker(ctx)
    rig.supervisor.apply_marker(ctx)
    assert rig.calls("mark") == before  # nothing changed: nothing sent


def test_a_failing_termctl_is_called_at_a_bounded_rate_and_logged_once(rig: MarkerRig) -> None:
    ctx = RunCtx(2, "/x")
    clock = Clock()
    rig.supervisor.clock = clock
    rig.relay(2, "needs-input", None)
    rig.fakebin.script("termctl", "mark", exit=12, error={"code": "locked", "message": "m"})
    for _ in range(50):  # fifty looks within one backoff period
        rig.supervisor.apply_marker(ctx)
        clock.now += 0.5
    assert rig.calls("mark") == 1
    for _ in range(3):  # the same failure after each backoff: still one warning
        clock.now += MARKER_BACKOFF + 1
        rig.supervisor.apply_marker(ctx)
    assert rig.calls("mark") == 4
    log = (Path(rig.supervisor.state_dir) / "log.jsonl").read_text()
    assert log.count("window marker failed") == 1


def test_a_pane_that_is_gone_is_marked_once_and_then_left_alone(rig: MarkerRig) -> None:
    ctx = RunCtx(2, "/x")
    rig.supervisor.clock = Clock()
    rig.relay(2, "needs-input", None)
    rig.fakebin.script("termctl", "mark", exit=10, error={"code": "not_found", "message": "m"})
    for _ in range(5):
        rig.supervisor.apply_marker(ctx)
    rig.relay(2, "working", "working")
    rig.supervisor.apply_marker(ctx)
    assert rig.calls("mark") == 1 and rig.calls("unmark") == 0


def test_no_marker_call_once_a_termination_signal_was_forwarded(rig: MarkerRig) -> None:
    rig.relay(2, "needs-input", None)
    rig.supervisor.terminated = True
    rig.supervisor.apply_marker(RunCtx(2, "/x"))
    assert rig.calls("mark") == 0


def test_a_state_that_means_nothing_changes_nothing(rig: MarkerRig) -> None:
    rig.relay(2, "starting", None)
    rig.supervisor.apply_marker(RunCtx(2, "/x"))
    rig.relay(2, "working", "working")
    rig.supervisor.apply_marker(RunCtx(2, "/x"))  # never marked: no unmark call either
    assert rig.calls("mark") == 0 and rig.calls("unmark") == 0


def test_only_the_current_runs_window_is_touched(rig: MarkerRig) -> None:
    rig.relay(1, "needs-input", None)  # a late state of the earlier run
    rig.supervisor.apply_marker(RunCtx(2, "/x"))
    assert rig.marker(rig.first) is None and rig.marker(rig.second) is None
    assert rig.calls("mark") == 0


def test_a_headless_supervisor_has_no_termctl(rig: MarkerRig) -> None:
    rig.supervisor.term = None
    rig.relay(2, "needs-input", None)
    rig.supervisor.apply_marker(RunCtx(2, "/x"))
    assert rig.calls("mark") == 0
