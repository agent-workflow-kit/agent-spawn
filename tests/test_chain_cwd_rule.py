"""The supervisor's automatic report resume runs all five pre-start checks: the records of the
cwd, the cwd rule or the throwaway check, and the holds (interface.md, Third-party runs).

End-to-end tests launch a real headless agent against the fakes; the in-process ones seed an
agent whose last run had a given cwd and drive the supervisor, so a cwd the launcher would have
refused can be exercised."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json
from tests.test_supervise_chain import (
    LAUNCH,
    chain_failure_of,
    real_starts,
    script_unreported_first,
)
from tests.test_supervise_inprocess import Rig
from tests.test_supervise_inprocess import rig_for as rig_for  # noqa: F401 - the fixture
from tests.test_worktrees_launch import fake_adopted, fake_throwaway, records, tree, wtctl_calls


@pytest.fixture(autouse=True)
def fakebin_on_path(fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch) -> None:
    """The supervisor of a real agent inherits the launcher's PATH, where `wtctl` resolves."""
    monkeypatch.setenv("PATH", fakebin.env()["PATH"])


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def log_events(state: Path) -> list[dict[str, Any]]:
    path = state / "log.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def park_failure(rig: Rig) -> dict[str, Any] | None:
    db = sqlite3.connect(f"file:{rig.state / 'state.db'}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT payload FROM events WHERE agent_id = ? AND event = 'post-park'",
            (rig.agent_id,),
        ).fetchall()
    finally:
        db.close()
    assert len(rows) == 1
    return json.loads(rows[0][0])["chain_failure"]  # type: ignore[no-any-return]


def starts(fakebin: Fakebin) -> list[list[str]]:
    return real_starts(fakebin)


def holds_of(rig: Rig) -> list[str]:
    from agent_spawn.store import Store

    with Store.open(rig.state, create=False) as store:
        return [h["record_id"] for h in store.get_holds(rig.agent_id)]


def seeded(rig_for: Callable[..., Rig], fakebin: Fakebin, cwd: str, **agent: Any) -> Rig:  # noqa: F811
    rig = rig_for(**agent)
    rig.put_run(1, cwd=cwd)
    script_unreported_first(fakebin)
    return rig


def carry_third_party(fakebin: Fakebin) -> None:
    """The launch's dry run answers as scripted before; every later one says third-party."""
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"], times=1)
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"profile": {"reads_third_party": True}},
    )


# --- end to end -------------------------------------------------------------------------------


def test_a_third_party_agent_at_a_throwaway_root_gets_its_report_resume_and_holds_nothing(
    env: Env, fakebin: Fakebin
) -> None:
    throwaway = fake_throwaway(fakebin)
    script_unreported_first(fakebin)
    carry_third_party(fakebin)  # the real claude-session carries the flag to a resume
    done = env.run(*LAUNCH, "--cwd", throwaway["path"], "--reads-third-party", "--wait")
    agent = result_json(done)["agent"]["agent_id"]
    assert done.returncode == 0, (done.stdout, chain_failure_of(env, agent))
    body = result_json(done)
    assert body["outcome"] == "done" and body["agent"]["holds"] == []
    assert len(starts(fakebin)) == 2  # the launch and the report resume
    assert wtctl_calls(fakebin, "adopt") == []
    assert wtctl_calls(fakebin, "list")  # the records of the cwd were looked up for the resume


def test_a_trusted_report_resume_in_a_throwaway_is_a_third_party_run_chain_failure(
    env: Env, fakebin: Fakebin
) -> None:
    throwaway = fake_throwaway(fakebin)
    script_unreported_first(fakebin)
    done = env.run(*LAUNCH, "--cwd", throwaway["path"], "--reads-third-party", "--wait")
    agent_id = result_json(done)["agent"]["agent_id"]
    failure = chain_failure_of(env, agent_id)
    assert failure is not None
    assert failure["tool"] == "agent-spawn" and failure["reason"] == "third_party_run"
    assert len(starts(fakebin)) == 1  # nothing was started
    row = env.agent(agent_id)
    assert row["park_reason"] == "unreported"
    log = log_events(env.state_dir)
    assert [r for r in log if r["event"] == "third_party_refused"]


# --- in process: the records of the cwd ---------------------------------------------------------


def test_a_trusted_resume_inside_a_throwaway_subdirectory_is_refused(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,  # noqa: F811
) -> None:
    throwaway = fake_throwaway(fakebin)
    sub = Path(throwaway["path"]) / "sub"
    sub.mkdir()
    rig = seeded(rig_for, fakebin, str(sub))
    rig.supervisor.run()
    failure = park_failure(rig)
    assert failure is not None
    assert (failure["tool"], failure["reason"]) == ("agent-spawn", "third_party_run")
    assert starts(fakebin) == []


def test_a_symlinked_recorded_cwd_is_judged_by_its_target(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    throwaway = fake_throwaway(fakebin)
    link = tmp_path / "innocent"
    link.symlink_to(throwaway["path"])
    rig = seeded(rig_for, fakebin, str(link))
    rig.supervisor.run()
    failure = park_failure(rig)
    assert failure is not None and failure["reason"] == "third_party_run"
    assert starts(fakebin) == []


def test_a_third_party_resume_below_a_throwaway_root_is_refused(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,  # noqa: F811
) -> None:
    throwaway = fake_throwaway(fakebin)
    sub = Path(throwaway["path"]) / "sub"
    sub.mkdir()
    rig = seeded(rig_for, fakebin, str(sub))
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"], times=0)
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"profile": {"reads_third_party": True}},
    )
    rig.supervisor.run()
    failure = park_failure(rig)
    assert failure is not None and failure["reason"] == "third_party_cwd"
    assert starts(fakebin) == []


def test_a_failed_records_lookup_is_a_wtctl_chain_failure_before_any_start(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    rig = seeded(rig_for, fakebin, str(tmp_path))
    fakebin.script("wtctl", "list", error={"code": "internal", "message": "x"}, exit=1)
    rig.supervisor.run()
    failure = park_failure(rig)
    assert failure is not None and failure["tool"] == "wtctl"
    assert starts(fakebin) == []
    assert wtctl_calls(fakebin, "adopt") == []


def test_an_unparseable_records_lookup_fails_closed(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    rig = seeded(rig_for, fakebin, str(tmp_path))
    fakebin.script("wtctl", "list", stdout="not json")
    rig.supervisor.run()
    failure = park_failure(rig)
    assert failure is not None and failure["tool"] == "wtctl"
    assert starts(fakebin) == []


# --- in process: wtctl availability -------------------------------------------------------------


def test_without_wtctl_a_trusted_resume_runs_and_a_third_party_one_needs_session_dir(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    rig = seeded(rig_for, fakebin, str(tmp_path))
    sup = rig.supervisor
    sup.config = sup.config.model_copy(update={"wtctl_cmd": None})
    sup.run()
    assert rig.agent()["park_reason"] == "done"
    assert len(starts(fakebin)) == 1
    assert wtctl_calls(fakebin) == []


def test_without_wtctl_a_third_party_resume_outside_session_dir_is_refused(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    rig = seeded(rig_for, fakebin, str(tmp_path))
    sup = rig.supervisor
    sup.config = sup.config.model_copy(update={"wtctl_cmd": None})
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"profile": {"reads_third_party": True}},
    )
    sup.run()
    failure = park_failure(rig)
    assert failure is not None and failure["reason"] == "third_party_cwd"
    assert wtctl_calls(fakebin) == [] and starts(fakebin) == []


def test_wtctl_missing_without_a_stored_path_is_tool_missing_before_any_start(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    rig = seeded(rig_for, fakebin, str(tmp_path))
    sup = rig.supervisor
    sup.config = sup.config.model_copy(update={"wtctl_cmd": [str(tmp_path / "gone" / "wtctl")]})
    sup.run()
    failure = park_failure(rig)
    assert failure is not None
    assert (failure["tool"], failure["reason"]) == ("wtctl", "tool_missing")
    assert starts(fakebin) == []


def test_the_stored_wtctl_path_is_the_fallback_when_config_cannot_resolve_it(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    rig = seeded(rig_for, fakebin, str(tmp_path), wtctl_path=str(fakebin.bin_dir / "wtctl"))
    sup = rig.supervisor
    sup.config = sup.config.model_copy(update={"wtctl_cmd": [str(tmp_path / "gone" / "wtctl")]})
    sup.run()
    assert rig.agent()["park_reason"] == "done"
    assert wtctl_calls(fakebin, "list")


def test_wtctl_is_not_needed_for_a_session_dir_run(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    rig = seeded(rig_for, fakebin, "session-dir")
    sup = rig.supervisor
    sup.config = sup.config.model_copy(update={"wtctl_cmd": [str(tmp_path / "gone" / "wtctl")]})
    sup.run()
    assert rig.agent()["park_reason"] == "done"
    assert wtctl_calls(fakebin) == []


def work_of(fakebin: Fakebin, rig: Rig) -> str:
    done = fakebin.run("claude-session", "status", rig.session, "--json")
    return os.path.join(json.loads(done.stdout)["csession_dir"], "work")


def test_a_third_party_agent_at_its_real_work_dir_needs_no_records_lookup(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    rig = rig_for()
    work = work_of(fakebin, rig)
    rig.put_run(1, cwd=work)
    script_unreported_first(fakebin)
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"profile": {"reads_third_party": True}},
    )
    sup = rig.supervisor
    sup.config = sup.config.model_copy(update={"wtctl_cmd": [str(tmp_path / "gone" / "wtctl")]})
    fakebin.script("wtctl", "list", error={"code": "internal", "message": "x"}, exit=1)
    sup.run()
    assert rig.agent()["park_reason"] == "done"
    assert len(starts(fakebin)) == 1
    assert wtctl_calls(fakebin, "list") == [] and wtctl_calls(fakebin, "adopt") == []


def test_an_unreadable_session_leaves_the_work_dir_unknown_and_fails_closed(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
) -> None:
    rig = rig_for()
    work = work_of(fakebin, rig)
    rig.put_run(1, cwd=work)
    script_unreported_first(fakebin)
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"profile": {"reads_third_party": True}},
    )
    fakebin.script("claude-session", "status", error={"code": "internal", "message": "x"}, exit=1)
    rig.supervisor.run()
    failure = park_failure(rig)
    assert failure is not None and failure["reason"] == "third_party_cwd"
    assert starts(fakebin) == []


# --- in process: the plan's cwd -----------------------------------------------------------------


def plan_cwd(fakebin: Fakebin, cwd: str, *, third_party: bool = False) -> None:
    merge: dict[str, Any] = {"cwd": cwd}
    if third_party:
        merge["profile"] = {"reads_third_party": True}
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"], merge=merge)


def test_a_trusted_run_whose_plan_cwd_is_a_throwaway_root_is_refused(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    throwaway = fake_throwaway(fakebin)
    plain = tmp_path / "plain"
    plain.mkdir()
    rig = seeded(rig_for, fakebin, str(plain))
    plan_cwd(fakebin, throwaway["path"])
    rig.supervisor.run()
    failure = park_failure(rig)
    assert failure is not None and failure["reason"] == "third_party_run"
    assert starts(fakebin) == []


def test_a_third_party_run_whose_plan_cwd_is_below_a_throwaway_root_is_refused(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    throwaway = fake_throwaway(fakebin)
    sub = Path(throwaway["path"]) / "sub"
    sub.mkdir()
    plain = tmp_path / "plain"
    plain.mkdir()
    rig = seeded(rig_for, fakebin, str(plain))
    plan_cwd(fakebin, str(sub), third_party=True)
    rig.supervisor.run()
    failure = park_failure(rig)
    assert failure is not None and failure["reason"] == "third_party_cwd"
    assert starts(fakebin) == []


def test_the_records_of_the_plans_cwd_are_held(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    root = tree(tmp_path)
    record = fake_adopted(fakebin, root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    rig = seeded(rig_for, fakebin, str(elsewhere))
    plan_cwd(fakebin, str(root))
    rig.supervisor.run()
    assert rig.agent()["park_reason"] == "done"
    assert holds_of(rig) == [record["id"]]
    assert len(starts(fakebin)) == 1


# --- in process: the holds ----------------------------------------------------------------------


def test_the_records_of_the_cwd_are_held_before_the_start(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    root = tree(tmp_path)
    inner = root / "pkg"
    outer_record = fake_adopted(fakebin, root)
    inner_record = fake_adopted(fakebin, inner)
    rig = seeded(rig_for, fakebin, str(inner))
    rig.supervisor.run()
    assert rig.agent()["park_reason"] == "done"
    assert set(holds_of(rig)) == {outer_record["id"], inner_record["id"]}
    live = records(fakebin)
    assert rig.agent_id in live[outer_record["id"]]["holders"]
    assert rig.agent_id in live[inner_record["id"]]["holders"]
    adopts = [a for a in wtctl_calls(fakebin, "adopt") if f"--holder={rig.agent_id}" in a]
    assert sorted(a[1] for a in adopts) == sorted([str(root.resolve()), str(inner.resolve())])
    assert rig.agent()["wt_record_id"] == inner_record["id"]  # the nearest
    assert len(starts(fakebin)) == 1


def test_a_hold_that_is_already_held_is_not_adopted_again(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    root = tree(tmp_path)
    rig = seeded(rig_for, fakebin, str(root))
    record = fake_adopted(fakebin, root, holder=rig.agent_id)
    from agent_spawn.store import Store

    with Store.open(rig.state, create=False) as store:
        with store.lease(f"agent:{rig.agent_id}") as lease:
            store.put_hold(lease, rig.agent_id, record["id"], record["path"], owned=False)
    before = len(wtctl_calls(fakebin, "adopt"))
    rig.supervisor.run()
    assert len(wtctl_calls(fakebin, "adopt")) == before
    assert holds_of(rig) == [record["id"]]


def test_a_failed_hold_is_a_wtctl_chain_failure_before_any_start(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    root = tree(tmp_path)
    fake_adopted(fakebin, root)
    rig = seeded(rig_for, fakebin, str(root))
    fakebin.script("wtctl", "adopt", error={"code": "busy", "message": "x"}, exit=12, times=1)
    rig.supervisor.run()
    failure = park_failure(rig)
    assert failure is not None
    assert (failure["tool"], failure["reason"]) == ("wtctl", "busy")
    assert starts(fakebin) == []
    assert holds_of(rig) == []


def test_a_run_at_a_throwaway_root_is_never_held(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,  # noqa: F811
) -> None:
    throwaway = fake_throwaway(fakebin)
    rig = seeded(rig_for, fakebin, throwaway["path"])
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"profile": {"reads_third_party": True}},
    )
    rig.supervisor.run()
    assert rig.agent()["park_reason"] == "done"
    assert holds_of(rig) == [] and wtctl_calls(fakebin, "adopt") == []


# --- in process: the classification check uses the records --------------------------------------


class _Captured(Exception):
    pass


def rule_of_the_chain_check(rig: Rig, work: str) -> Callable[[bool, str | None], bool]:
    """The ``rule_allows`` callback the chain hands to the classification check."""
    from agent_spawn import chain
    from agent_spawn.session_backend import StartResult

    caught: list[Callable[[bool, str | None], bool]] = []

    class Stub:
        actor = "auto:agent-spawn"

        def check_classification(self, *args: Any, **kwargs: Any) -> Any:
            caught.append(kwargs["rule_allows"])
            raise _Captured

    sup = rig.supervisor
    started = StartResult("cs-abcdefgh", os.path.dirname(work), 2, (), None, (), (), False, None)
    classification = chain.third_party.Classification(True, "k", None, None, None, None)
    wt = chain._Wtctl(sup, Stub(), rig.agent())  # pyright: ignore[reportPrivateUsage,reportArgumentType]
    with pytest.raises(_Captured):
        chain._classification_check(  # pyright: ignore[reportPrivateUsage]
            sup,
            Stub(),  # pyright: ignore[reportArgumentType]
            started,
            classification,
            work,
            None,
            "tok",
            wt,
        )
    return caught[0]


def test_the_post_start_rule_is_judged_by_the_records(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    throwaway = fake_throwaway(fakebin)
    rig = rig_for()
    rig.supervisor.store = None
    from agent_spawn.store import Store

    rig.supervisor.store = Store.open(rig.state, create=False)
    try:
        rule = rule_of_the_chain_check(rig, str(tmp_path / "cs" / "work"))
        assert rule(True, throwaway["path"]) is True  # third-party at a throwaway root
        work = str(tmp_path / "cs" / "work")
        assert rule(True, work) is True  # the session's own work/ lies in no record
        assert rule(False, throwaway["path"]) is False  # trusted inside a throwaway
        assert rule(True, str(Path(throwaway["path"]) / "sub")) is False
        assert rule(True, str(tmp_path)) is False  # outside every record
        assert len(wtctl_calls(fakebin, "list")) == 1  # one lookup for the whole check
    finally:
        rig.supervisor.store.close()


def test_a_failed_lookup_in_the_post_start_rule_raises_so_the_run_is_canceled(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,  # noqa: F811
) -> None:
    from agent_spawn.output import CliError
    from agent_spawn.store import Store

    rig = rig_for()
    rig.supervisor.store = Store.open(rig.state, create=False)
    try:
        rule = rule_of_the_chain_check(rig, str(tmp_path / "cs" / "work"))
        fakebin.script("wtctl", "list", error={"code": "internal", "message": "x"}, exit=1)
        with pytest.raises(CliError):
            rule(True, str(tmp_path))
    finally:
        rig.supervisor.store.close()


# --- end to end: the post-start check through the real session backend --------------------------


def aftermath(env: Env, fakebin: Fakebin, *args: str) -> tuple[str, dict[str, Any]]:
    done = env.run(*LAUNCH, *args, "--wait")
    agent_id = result_json(done)["agent"]["agent_id"]
    failure = chain_failure_of(env, agent_id)
    assert failure is not None, (done.stdout, done.stderr)
    assert len(starts(fakebin)) == 2  # the launch and the (canceled) report resume
    assert [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]
    assert len([a for a in fakebin.argvs("claude-session") if a[0] == "exec"]) == 1
    return agent_id, failure


def test_a_third_party_run_started_outside_every_record_is_canceled_as_a_mismatch(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    throwaway = fake_throwaway(fakebin)
    script_unreported_first(fakebin)
    carry_third_party(fakebin)
    fakebin.script("claude-session", "status", times=2)  # the launcher's check, the pre-start read
    fakebin.script(
        "claude-session",
        "status",
        merge={"run": {"reads_third_party": True, "cwd": str(tmp_path)}, "tainted_by_run": 2},
    )
    _, failure = aftermath(env, fakebin, "--cwd", throwaway["path"], "--reads-third-party")
    assert (failure["tool"], failure["reason"]) == ("agent-spawn", "classification_mismatch")


def test_a_failed_records_lookup_in_the_post_start_check_cancels_the_run_as_unknown(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    throwaway = fake_throwaway(fakebin)
    script_unreported_first(fakebin)
    carry_third_party(fakebin)
    fakebin.script("claude-session", "status", times=2)
    fakebin.script(
        "claude-session",
        "status",
        merge={"run": {"reads_third_party": True, "cwd": str(tmp_path)}, "tainted_by_run": 2},
    )
    # the launcher's two lookups, the chain's pre-start and holds lookups; the fifth is the check's
    fakebin.script("wtctl", "list", times=4)
    fakebin.script("wtctl", "list", error={"code": "internal", "message": "x"}, exit=1)
    _, failure = aftermath(env, fakebin, "--cwd", throwaway["path"], "--reads-third-party")
    assert (failure["tool"], failure["reason"]) == ("agent-spawn", "classification_unknown")


def test_a_run_that_turned_third_party_without_a_cwd_is_canceled_as_unknown(
    env: Env, fakebin: Fakebin
) -> None:
    script_unreported_first(fakebin)
    fakebin.script("claude-session", "status", times=1)
    fakebin.script(
        "claude-session",
        "status",
        merge={"run": {"reads_third_party": True, "cwd": None}, "tainted_by_run": 2},
    )
    _, failure = aftermath(env, fakebin)
    assert (failure["tool"], failure["reason"]) == ("agent-spawn", "classification_unknown")
