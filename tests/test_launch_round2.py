"""Round 2 review of the headless launch: park order, handler restore, interrupt kinds."""

from __future__ import annotations

import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent_spawn import cli, launch
from agent_spawn import gateway as gw
from agent_spawn.ids import new_agent_id
from agent_spawn.output import CliError
from agent_spawn.store import Store, StoreError
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.test_gateway_run import run
from tests.test_launch_review import SpawningGateway

SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


class RecordingBackend:
    def __init__(self) -> None:
        self.stopped: list[str] = []

    def stop(self, session: str) -> None:
        self.stopped.append(session)


def insert(store: Store, status: str) -> str:
    agent_id = new_agent_id(store)
    store.insert_agent(
        agent_id,
        {"status": status, "name": "agent test", "mode": "headless", "csession_id": "cs-abcdefgh"},
    )
    return agent_id


def fake_launch(store: Store, **extra: Any) -> Any:
    return SimpleNamespace(
        store=store,
        backend=RecordingBackend(),
        log=lambda *a, **k: None,
        decision_ids=[],
        **extra,
    )


# --- 1: park before stopping ------------------------------------------------------------------


def test_a_stale_launching_row_does_not_stop_a_live_agents_session(
    isolation_env: Isolation, tmp_path: Path
) -> None:
    with Store.open(tmp_path / "state") as store:
        agent_id = insert(store, "live")  # the supervisor made it live after the stale read
        fake = fake_launch(store)
        stale = {"agent_id": agent_id, "csession_id": "cs-abcdefgh", "status": "launching"}
        launch.Launch.park_unsupervised(fake, stale)
        assert fake.backend.stopped == []
        assert (store.get_agent(agent_id) or {})["status"] == "live"


def test_a_launching_agent_is_parked_then_its_session_stopped(
    isolation_env: Isolation, tmp_path: Path
) -> None:
    with Store.open(tmp_path / "state") as store:
        agent_id = insert(store, "launching")
        fake = fake_launch(store)
        row = {"agent_id": agent_id, "csession_id": "cs-abcdefgh"}
        with pytest.raises(CliError) as caught:
            launch.Launch.park_unsupervised(fake, row)
        assert caught.value.code == "launch_failed"
        assert fake.backend.stopped == ["cs-abcdefgh"]
        assert (store.get_agent(agent_id) or {})["status"] == "parked"


# --- 2: handlers are restored for in-process callers --------------------------------------------


def test_signal_handlers_are_restored_after_an_in_process_launch(
    fakebin: Fakebin,
    isolation_env: Isolation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = {s: signal.getsignal(s) for s in SIGNALS}
    try:
        monkeypatch.setattr("agent_spawn.gateway._default", SpawningGateway(fakebin))
        monkeypatch.setenv("PATH", fakebin.env()["PATH"])

        def failing(self: Store, lease: Any, agent_id: str, changes: Any, **kw: Any) -> Any:
            raise StoreError("lease timed out")

        monkeypatch.setattr(Store, "update_agent", failing)
        inv = cli.parse_invocation(
            ["launch", "--mode", "headless", "--skill", "do-it", "--json", "--prompt", "p"]
        )
        with pytest.raises(CliError):
            launch.launch(inv)
        assert {s: signal.getsignal(s) for s in SIGNALS} == saved
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def test_handlers_kept_for_the_command_line_are_restored_on_request() -> None:
    saved = {s: signal.getsignal(s) for s in SIGNALS}
    try:
        restore = launch.procs.install_signal_handlers({signal.SIGTERM: signal.SIG_IGN})
        launch._pending_restore = restore  # pyright: ignore[reportPrivateUsage]
        launch.restore_signal_handlers()
        launch.restore_signal_handlers()  # idempotent
        assert {s: signal.getsignal(s) for s in SIGNALS} == saved
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


# --- 3: only a launcher interrupt gets the polite SIGTERM ---------------------------------------


def _alarm_raising(
    exc: BaseException, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[int]:
    polite: list[int] = []

    def stop(proc: Any) -> None:
        polite.append(proc.pid)
        proc.kill()
        proc.wait()

    monkeypatch.setattr(gw, "_stop_politely", stop)

    def fire(signum: int, frame: object) -> None:
        raise exc

    old = signal.signal(signal.SIGALRM, fire)
    try:
        signal.setitimer(signal.ITIMER_REAL, 1.0)
        with pytest.raises(type(exc)):
            run(tmp_path, "import time; time.sleep(60)", timeout=30)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    return polite


def test_an_ordinary_exception_kills_the_child_without_the_polite_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _alarm_raising(OSError("boom"), tmp_path, monkeypatch) == []


def test_a_launcher_interrupt_gets_the_polite_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert len(_alarm_raising(launch.LaunchInterrupted(signal.SIGTERM), tmp_path, monkeypatch)) == 1


# --- 4: a signal during placement cleanup is recorded, not raised -------------------------------


def test_a_signal_during_the_placement_cleanup_does_not_abort_it(
    isolation_env: Isolation, tmp_path: Path
) -> None:
    with Store.open(tmp_path / "state") as store:
        agent_id = insert(store, "launching")
        sig = launch.SignalState()
        detached = SimpleNamespace(abort=lambda: sig.handle(signal.SIGTERM, None))
        fake = fake_launch(store, sig=sig, agent_id=agent_id, detached=detached)
        with pytest.raises(CliError) as caught:
            launch.Launch.placement_failed(
                fake, cast(Any, SimpleNamespace(csession_id="cs-abcdefgh")), OSError("x")
            )
        assert caught.value.code == "launch_failed"
        assert sig.take() == [signal.SIGTERM]
        assert (store.get_agent(agent_id) or {})["status"] == "parked"
        assert fake.backend.stopped == ["cs-abcdefgh"]
