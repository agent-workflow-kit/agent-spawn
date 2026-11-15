"""``list``, ``show``, ``ack`` and ``label add|remove`` against a real database and the fakes."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from agent_spawn.agent_view import outcome_object
from agent_spawn.session_backend import RunOutcome
from agent_spawn.store import Store
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.inspect_support import A1, A2, A3, SESSION, labels_of, run_inprocess, seed_agent
from tests.launch_support import Env

LIST = "agent-spawn/cli/list.json"
SHOW = "agent-spawn/cli/show.json"
ACK = "agent-spawn/cli/ack.json"
ADD = "agent-spawn/cli/label-add.json"
REMOVE = "agent-spawn/cli/label-remove.json"
ERROR = "agent-spawn/cli/error.json"


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def out(done: Any, schema: str, code: int = 0) -> Any:
    body = json.loads(done.stdout)
    assert done.returncode == code, (done.stdout, done.stderr)
    schema_registry.validate(body, schema)
    return body


def ids(body: Any) -> list[str]:
    return [a["agent_id"] for a in body["agents"]]


# --- list --------------------------------------------------------------------------------------


def test_list_without_a_database_is_empty_and_creates_nothing(env: Env) -> None:
    body = out(env.run("list", "--json"), LIST)
    assert body == {"agents": []}
    assert not env.state_dir.exists()


def test_list_orders_by_created_descending_then_id(env: Env) -> None:
    seed_agent(env, A1, created=100.0)
    seed_agent(env, A3, created=200.0)
    seed_agent(env, A2, created=200.0)
    assert ids(out(env.run("list", "--json"), LIST)) == [A2, A3, A1]


def test_list_hides_torn_down_unless_status_names_it(env: Env) -> None:
    seed_agent(env, A1, status="torn_down", torn_down_at=10.0)
    seed_agent(env, A2)
    assert ids(out(env.run("list", "--json"), LIST)) == [A2]
    assert ids(out(env.run("list", "--status", "torn_down", "--json"), LIST)) == [A1]
    both = out(env.run("list", "--status", "torn_down", "--status", "parked", "--json"), LIST)
    assert sorted(ids(both)) == [A1, A2]


def test_list_labels_are_anded_and_a_multiset(env: Env) -> None:
    seed_agent(env, A1, labels=["ticket=T-1", "owner=x", "job=a", "job=b"])
    seed_agent(env, A2, labels=["ticket=T-1"])
    one = out(env.run("list", "--label", "ticket=T-1", "--json"), LIST)
    assert sorted(ids(one)) == [A1, A2]
    two = out(env.run("list", "--label", "ticket=T-1", "--label", "owner=x", "--json"), LIST)
    assert ids(two) == [A1]
    jobs = out(env.run("list", "--label", "job=a", "--label", "job=b", "--json"), LIST)
    assert ids(jobs) == [A1]
    assert ids(out(env.run("list", "--label", "job=zzz", "--json"), LIST)) == []


@pytest.mark.parametrize("key", ["agent", "kind"])
def test_list_rejects_reserved_label_keys(env: Env, key: str) -> None:
    done = env.run("list", "--label", f"{key}=x", "--json")
    assert out(done, ERROR, 2)["error"]["code"] == "usage"


def test_list_filters(env: Env) -> None:
    seed_agent(env, A1, launch_kind="implement", run_kind="review", repo="h/o/r", mode="headless")
    seed_agent(
        env, A2, launch_kind="respond", mode="interactive", placement="window",
        needs_attention="blocked", session_state="needs-input", forked_from=A1,
    )  # fmt: skip
    seed_agent(env, A3, status="live", session_state="working", repo="h/o/other")

    def pick(*args: str) -> list[str]:
        return sorted(ids(out(env.run("list", *args, "--json"), LIST)))

    assert pick("--needs-attention") == [A2]
    assert pick("--kind", "review") == [A1]  # the last run's kind
    assert pick("--kind", "implement") == [A1]  # the launch kind
    assert pick("--repo", "h/o/r") == [A1]
    assert pick("--mode", "interactive") == [A2]
    assert pick("--forked-from", A1) == [A2]
    assert pick("--forked-from", A1[3:7]) == [A2]  # a prefix resolves
    assert pick("--status", "live") == [A3]
    assert pick("--session-state", "working", "--session-state", "needs-input") == [A2, A3]
    assert pick("--session-state", "working") == [A3]


def test_list_since_matches_updated(env: Env) -> None:
    seed_agent(env, A1, created=1000.0, updated=1000.0)
    seed_agent(env, A2)  # updated now
    assert ids(out(env.run("list", "--since", "1h", "--json"), LIST)) == [A2]
    assert sorted(ids(out(env.run("list", "--since", "1970-01-01", "--json"), LIST))) == [A1, A2]


def test_list_unknown_forked_from_and_bad_choices(env: Env) -> None:
    seed_agent(env, A1)
    assert out(env.run("list", "--forked-from", "zzzz", "--json"), ERROR, 10)  # a prefix
    assert out(env.run("list", "--status", "nope", "--json"), ERROR, 2)["error"]["code"] == "usage"
    done = env.run("list", "--session-state", "nope", "--json")
    assert out(done, ERROR, 2)["error"]["code"] == "usage"


def test_list_never_calls_claude_session_or_delivers_hooks(env: Env, fakebin: Fakebin) -> None:
    seed_agent(env, A1, status="live", csession_id=SESSION, session_state="working")
    body = out(env.run("list", "--json"), LIST)
    assert body["agents"][0]["session_state"] == "working"
    assert fakebin.argvs("claude-session") == []


def test_list_never_leaks_internal_columns(env: Env) -> None:
    seed_agent(
        env, A1, settings={"secret": "TOPSECRET"}, hooks=[{"x": "TOPSECRET"}], decision_ids=["d"],
    )  # fmt: skip
    done = env.run("list", "--json")
    assert "TOPSECRET" not in done.stdout and "TOPSECRET" not in done.stderr
    human = env.run("list")
    assert human.returncode == 0 and A1 in human.stdout


def test_list_human_output_escapes_names(env: Env) -> None:
    seed_agent(env, A1, name="evil\x1b[31mname")
    human = env.run("list")
    assert "\x1b" not in human.stdout


# --- show --------------------------------------------------------------------------------------


def test_show_without_a_database_is_not_found(env: Env) -> None:
    body = out(env.run("show", A1, "--json"), ERROR, 10)
    assert body["error"]["code"] == "not_found"
    assert not env.state_dir.exists()


def test_show_a_parked_agent_with_runs(env: Env, fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=True, outcome="done")
    outcome = outcome_object(
        A1, RunOutcome(SESSION, 1, "session", "killed", "stopped", 15, "killed", None, "status")
    )
    seed_agent(
        env, A1, csession_id=SESSION, run=1,
        claude_session_path=str(fakebin.bin_dir / "claude-session"),
        runs=[
            {"mode": "headless", "run_kind": "implement", "outcome": outcome, "ended_at": 5.0,
             "first_working_at": 3.0, "promoted": False, "auto_resumed": True},
            {"mode": "headless"},
        ],
    )  # fmt: skip
    body = out(env.run("show", A1, "--json"), SHOW)
    assert [r["run"] for r in body["runs"]] == [1, 2]
    assert body["runs"][0]["outcome"] == outcome
    assert body["runs"][0]["auto_resumed"] is True
    assert body["runs"][0]["first_working_at"] == "1970-01-01T00:00:03.000Z"
    assert body["runs"][1]["outcome"] is None
    assert body["session"]["state"] == "exited"
    assert body["session"]["tainted"] is False
    assert body["supervisor"] is None
    assert body["repo_config_changed"] is None
    fakebin.assert_argv("claude-session", ["status", SESSION, "--json"])


def test_show_prefix_and_ambiguity(env: Env) -> None:
    seed_agent(env, "ag-abcdefgh")
    seed_agent(env, "ag-abcdzzzz")
    assert out(env.run("show", "ag-abcd", "--json"), ERROR, 21)["error"]["candidates"] == [
        "ag-abcdefgh",
        "ag-abcdzzzz",
    ]
    assert out(env.run("show", "abcde", "--json"), SHOW)["agent"]["agent_id"] == "ag-abcdefgh"
    assert out(env.run("show", "ag-ab", "--json"), ERROR, 2)["error"]["code"] == "usage"


def test_show_a_tombstone_by_full_id_only(env: Env, fakebin: Fakebin) -> None:
    seed_agent(env, A1, status="torn_down", csession_id=SESSION, torn_down_at=10.0)
    body = out(env.run("show", A1, "--json"), SHOW)
    assert body["session"] is None
    assert body["supervisor"] is None
    assert body["repo_config_changed"] is None
    assert fakebin.argvs("claude-session") == []
    assert out(env.run("show", A1[:7], "--json"), ERROR, 10)["error"]["code"] == "not_found"


def test_show_session_not_found_is_a_warning_and_null(env: Env, fakebin: Fakebin) -> None:
    seed_agent(
        env, A1, status="live", csession_id="cs-zzzzzzzz",
        claude_session_path=str(fakebin.bin_dir / "claude-session"),
    )  # fmt: skip
    body = out(env.run("show", A1, "--json"), SHOW)
    assert body["session"] is None
    assert body["agent"]["status"] == "live"  # TODO(#31) parks it; not this issue
    assert any("not found" in w for w in body["warnings"])


def test_show_supervisor_alive_and_dead(env: Env) -> None:
    with Store.open(env.state_dir) as store:
        me = store.holder_identity()
    seed_agent(env, A1, status="live", supervisor=me)
    body = out(env.run("show", A1, "--json"), SHOW)
    assert body["supervisor"]["pid"] == me.pid
    assert body["supervisor"]["exec_pid"] is None
    # `show` ran in a child process: the recorded PID is the test's (alive); a dead identity:
    from agent_spawn.procs import ProcessIdentity

    dead = ProcessIdentity(pid=2**22 - 3, start_time=1, boot_id=me.boot_id)
    seed_agent(env, A2, status="live", supervisor=dead, exec_proc=dead)
    body = out(env.run("show", A2, "--json"), SHOW)
    assert body["supervisor"] == {"pid": dead.pid, "alive": False, "exec_pid": dead.pid}


def test_show_third_party_run_without_wtctl(env: Env, isolation_env: Isolation) -> None:
    cfg = isolation_env.config / "agent-spawn"
    cfg.mkdir(parents=True)
    (cfg / "config.yaml").write_text("wtctl_cmd: null\n")
    seed_agent(env, A1, cwd="/work/x")
    seed_agent(env, A2)
    assert out(env.run("show", A1, "--json"), SHOW)["third_party_run"] is False
    assert out(env.run("show", A2, "--json"), SHOW)["third_party_run"] is None  # cwd unknown


def test_show_leaks_no_prompt_or_secret_text(env: Env, fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION)
    seed_agent(
        env, A1, csession_id=SESSION, settings={"p": "PROMPTTEXT"}, hooks=["PROMPTTEXT"],
        claude_session_path=str(fakebin.bin_dir / "claude-session"),
    )  # fmt: skip
    done = env.run("show", A1, "--json")
    assert "PROMPTTEXT" not in done.stdout
    human = env.run("show", A1)
    assert human.returncode == 0 and "PROMPTTEXT" not in human.stdout


# --- ack ---------------------------------------------------------------------------------------


def hook_events(env: Env, agent_id: str) -> list[dict[str, Any]]:
    with env.store() as store:
        return store.pending_events(agent_id)


def test_ack_clears_needs_attention_and_records_an_event(env: Env) -> None:
    seed_agent(env, A1, status="live", needs_attention="blocked", supervisor_lost=True)
    body = out(env.run("ack", A1, "--json"), ACK)
    assert body["changed"] is True
    assert body["agent"]["needs_attention"] is None
    assert body["agent"]["supervisor_lost"] is True  # ack never clears supervisor_lost
    assert env.agent(A1)["needs_attention"] is None
    log = (env.state_dir / "log.jsonl").read_text()
    assert '"event": "ack"' in log or '"event":"ack"' in log


def test_ack_with_nothing_to_clear_is_a_no_op(env: Env) -> None:
    row = seed_agent(env, A1)
    body = out(env.run("ack", A1, "--json"), ACK)
    assert body["changed"] is False
    assert env.agent(A1)["updated"] == row["updated"]
    assert hook_events(env, A1) == []


def test_ack_queues_post_state_change_when_no_hook_delivers(env: Env) -> None:
    # with no configured hooks the event is queued and delivered (marked) by the command
    seed_agent(env, A1, needs_attention="failed")
    out(env.run("ack", A1, "--json"), ACK)
    with env.store() as store:
        rows = store._conn.execute(  # noqa: SLF001
            "SELECT event FROM events WHERE agent_id = ?", (A1,)
        ).fetchall()
    assert [r[0] for r in rows] == ["post-state-change"]


def test_ack_torn_down_and_unknown(env: Env) -> None:
    seed_agent(env, A1, status="torn_down", torn_down_at=10.0)
    body = out(env.run("ack", A1, "--json"), ERROR, 10)
    assert body["error"]["details"]["status"] == "torn_down"
    assert out(env.run("ack", "ag-qqqqqqqq", "--json"), ERROR, 10)["error"]["code"] == "not_found"


def test_ack_without_a_database_creates_none(env: Env) -> None:
    assert out(env.run("ack", A1, "--json"), ERROR, 10)
    assert not env.state_dir.exists()


# --- label add / remove ------------------------------------------------------------------------


def test_label_add_updates_stored_labels_and_mirrors_to_the_session(
    env: Env, fakebin: Fakebin
) -> None:
    fakebin.seed_session(SESSION, labels=[f"agent={A1}"])
    seed_agent(
        env, A1, labels=["a=1"], csession_id=SESSION,
        claude_session_path=str(fakebin.bin_dir / "claude-session"),
    )  # fmt: skip
    body = out(env.run("label", "add", A1, "pr=h/o/r#7", "a=1", "--json"), ADD)
    assert body["added"] == ["pr=h/o/r#7"]
    assert body["labels"] == ["a=1", "pr=h/o/r#7"]
    assert body["changed"] is True and body["dry_run"] is False and body["warnings"] == []
    fakebin.assert_argv("claude-session", ["label", "add", SESSION, "pr=h/o/r#7", "--json"])
    with env.store() as store:
        assert store.mirror_pending(A1) == []
        assert store.get_labels(A1).texts() == ["a=1", "pr=h/o/r#7"]


def test_label_add_is_a_multiset_no_op(env: Env, fakebin: Fakebin) -> None:
    seed_agent(env, A1, labels=["a=1"], csession_id=SESSION)
    body = out(env.run("label", "add", A1, "a=1", "--json"), ADD)
    assert body["changed"] is False and body["added"] == []
    assert fakebin.argvs("claude-session") == []
    same = out(env.run("label", "add", A1, "a=2", "a=2", "--json"), ADD)
    assert same["added"] == ["a=2"]
    assert same["labels"] == ["a=1", "a=2"]


def test_label_remove_and_absent_pair(env: Env, fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION, labels=["a=1"])
    seed_agent(
        env, A1, labels=["a=1", "b=2"], csession_id=SESSION,
        claude_session_path=str(fakebin.bin_dir / "claude-session"),
    )  # fmt: skip
    body = out(env.run("label", "remove", A1, "a=1", "zz=9", "--json"), REMOVE)
    assert body["removed"] == ["a=1"] and body["labels"] == ["b=2"] and body["changed"]
    fakebin.assert_argv("claude-session", ["label", "remove", SESSION, "a=1", "--json"])
    noop = out(env.run("label", "remove", A1, "a=1", "--json"), REMOVE)
    assert noop["changed"] is False and noop["removed"] == []


def test_label_remove_has_no_cap(env: Env) -> None:
    seed_agent(env, A1, labels=[f"k{i}=v" for i in range(64)])
    gone = [f"k{i}=v" for i in range(3)] + [f"zz{i}=v" for i in range(70)]
    body = out(env.run("label", "remove", A1, *gone, "--json"), REMOVE)
    assert len(body["removed"]) == 3 and len(body["labels"]) == 61


def test_label_add_cap_is_checked_against_the_stored_set(env: Env) -> None:
    seed_agent(env, A1, labels=[f"k{i}=v" for i in range(63)])
    body = out(env.run("label", "add", A1, "x=1", "y=2", "--json"), ERROR, 2)
    assert body["error"]["code"] == "usage"
    assert len(labels_of(env, A1)) == 63  # nothing was written
    ok = out(env.run("label", "add", A1, "x=1", "--json"), ADD)
    assert len(ok["labels"]) == 64
    job = out(env.run("label", "add", A1, "job=late", "--json"), ERROR, 2)  # no job= exemption
    assert job["error"]["code"] == "usage"
    again = out(env.run("label", "add", A1, "x=1", "--json"), ADD)  # a present pair still no-ops
    assert again["changed"] is False


@pytest.mark.parametrize("verb", ["add", "remove"])
@pytest.mark.parametrize("pair", ["agent=x", "kind=x", "bad key=1", "a", "A=1", "a=b c"])
def test_label_edit_rejects_reserved_and_malformed_pairs(env: Env, verb: str, pair: str) -> None:
    seed_agent(env, A1)
    done = env.run("label", verb, A1, pair, "--json")
    assert out(done, ERROR, 2)["error"]["code"] == "usage"


def test_label_needs_at_least_one_pair(env: Env) -> None:
    seed_agent(env, A1)
    assert out(env.run("label", "add", A1, "--json"), ERROR, 2)


def test_label_dry_run_changes_nothing_and_calls_nothing(env: Env, fakebin: Fakebin) -> None:
    seed_agent(env, A1, labels=["a=1"], csession_id=SESSION)
    body = out(env.run("label", "add", A1, "b=2", "--dry-run", "--json"), ADD)
    assert body["dry_run"] is True and body["added"] == ["b=2"] and body["labels"] == ["a=1", "b=2"]
    assert labels_of(env, A1) == ["a=1"]
    rem = out(env.run("label", "remove", A1, "a=1", "--dry-run", "--json"), REMOVE)
    assert rem["dry_run"] is True and rem["removed"] == ["a=1"] and rem["labels"] == []
    assert labels_of(env, A1) == ["a=1"]
    assert fakebin.argvs("claude-session") == []
    seed_agent(env, A2, labels=[f"k{i}=v" for i in range(64)])
    over = out(env.run("label", "add", A2, "x=1", "--dry-run", "--json"), ERROR, 2)
    assert over["error"]["code"] == "usage"


def test_label_mirror_failure_is_a_warning_and_stays_queued(env: Env, fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION)
    fakebin.script(
        "claude-session", "label add", exit=12, error={"code": "failed", "message": "boom"}
    )
    seed_agent(
        env, A1, csession_id=SESSION, claude_session_path=str(fakebin.bin_dir / "claude-session")
    )
    body = out(env.run("label", "add", A1, "a=1", "--json"), ADD)
    assert body["changed"] is True and len(body["warnings"]) == 1
    assert "boom" not in body["warnings"][0]  # lower-tool text is not echoed into warnings
    with env.store() as store:
        assert store.get_labels(A1).texts() == ["a=1"]
        pending = store.mirror_pending(A1)
    assert [(str(p), op) for p, op in pending] == [("a=1", "add")]


def test_label_without_a_session_queues_without_a_warning(env: Env, fakebin: Fakebin) -> None:
    seed_agent(env, A1)
    body = out(env.run("label", "add", A1, "a=1", "--json"), ADD)
    assert body["warnings"] == [] and body["labels"] == ["a=1"]
    assert fakebin.argvs("claude-session") == []
    with env.store() as store:
        assert [(str(p), op) for p, op in store.mirror_pending(A1)] == [("a=1", "add")]


def test_label_on_a_torn_down_agent_is_not_found(env: Env) -> None:
    seed_agent(env, A1, status="torn_down", torn_down_at=10.0)
    for args in (("add", A1, "a=1"), ("remove", A1, "a=1")):
        body = out(env.run("label", *args, "--json"), ERROR, 10)
        assert body["error"]["details"]["status"] == "torn_down"


def test_label_without_a_database_creates_none(env: Env) -> None:
    assert out(env.run("label", "add", A1, "a=1", "--json"), ERROR, 10)
    assert not env.state_dir.exists()


def test_label_changes_are_logged(env: Env) -> None:
    seed_agent(env, A1)
    out(env.run("label", "add", A1, "a=1", "--json"), ADD)
    records = [json.loads(line) for line in (env.state_dir / "log.jsonl").read_text().splitlines()]
    change = [r for r in records if r["event"] == "label_change"]
    assert change and change[0]["details"] == {"op": "add", "labels": ["a=1"]}
    assert change[0]["actor"].startswith("auto:")


def test_concurrent_label_adds_keep_every_pair(env: Env) -> None:
    import subprocess

    seed_agent(env, A1)
    procs = [
        subprocess.Popen(
            env.argv("label", "add", A1, f"k{i}=v", "--json"),
            env=env.env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        )
        for i in range(4)
    ]  # fmt: skip
    for proc in procs:  # short-lived children of this test; each is waited for
        proc.communicate(timeout=60)
        assert proc.returncode == 0
    assert sorted(labels_of(env, A1)) == [f"k{i}=v" for i in range(4)]


def test_corrupt_database_is_an_expected_failure(env: Env) -> None:
    seed_agent(env, A1)
    db = env.state_dir / "state.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    db.write_bytes(b"not a database" * 100)
    done = env.run("list", "--json")
    assert done.returncode == 12
    schema_registry.validate(json.loads(done.stdout), ERROR)


# --- review round 1 ----------------------------------------------------------------------------


def test_list_forked_from_a_full_id_without_a_record_lists_its_children(env: Env) -> None:
    # R10: the parent's tombstone may be deleted; its children stay listable
    seed_agent(env, A2, forked_from=A1)
    seed_agent(env, A3)
    assert ids(out(env.run("list", "--forked-from", A1, "--json"), LIST)) == [A2]
    assert ids(out(env.run("list", "--forked-from", "ag-zzzzzzzz", "--json"), LIST)) == []
    assert out(env.run("list", "--forked-from", "ag-zz", "--json"), ERROR, 2)


@pytest.mark.parametrize("bad", [1e300, "abc"])
@pytest.mark.parametrize("command", ["list", "show"])
def test_a_corrupt_row_is_exit_12_not_internal(env: Env, command: str, bad: Any) -> None:
    # R14
    seed_agent(env, A1)
    db = sqlite3.connect(env.state_dir / "state.db")
    with db:
        db.execute("UPDATE agents SET updated = ?", (bad,))
    db.close()
    args = [command] if command == "list" else [command, A1]
    body = out(env.run(*args, "--json"), ERROR, 12)
    assert body["error"]["code"] == "failed"
    assert body["error"]["message"].startswith("state.db row unusable")


def test_show_a_session_answering_for_another_id_is_null_with_a_warning(
    env: Env, fakebin: Fakebin
) -> None:
    # R13
    fakebin.seed_session(SESSION)
    fakebin.script("claude-session", "status", merge={"csession_id": "cs-zzzzzzzz"})
    seed_agent(
        env, A1, status="live", csession_id=SESSION,
        claude_session_path=str(fakebin.bin_dir / "claude-session"),
    )  # fmt: skip
    body = out(env.run("show", A1, "--json"), SHOW)
    assert body["session"] is None
    assert any("another session" in w for w in body["warnings"])


def test_show_an_unreadable_supervisor_entry_fails_closed(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # R2: never "dead" when the process entry cannot be read
    from agent_spawn import procs

    with Store.open(env.state_dir) as store:
        me = store.holder_identity()
    seed_agent(env, A1, status="live", supervisor=me)

    def unreadable(*_a: Any, **_k: Any) -> bool:
        raise procs.ProcessIdentityError("/proc/123/stat said SECRETPROC")

    monkeypatch.setattr(procs, "is_alive", unreadable)
    code, stdout, stderr = run_inprocess(env, monkeypatch, capsys, "show", A1, "--json")
    body = json.loads(stdout)
    assert code == 12 and body["error"]["code"] == "failed"
    schema_registry.validate(body, ERROR)
    assert "SECRETPROC" not in stdout + stderr and "alive" not in stdout


def test_show_third_party_run_is_null_with_a_warning_when_wtctl_is_set(env: Env) -> None:
    # R6d (until #21 looks the records up)
    seed_agent(env, A1, cwd="/work/x")
    body = out(env.run("show", A1, "--json"), SHOW)
    assert body["third_party_run"] is None
    assert any("throwaway records" in w for w in body["warnings"])


def test_ack_on_a_held_lease_is_retryable_exit_12(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # R6a: leases are not reentrant, so a lease held by this very process fails at once
    seed_agent(env, A1, needs_attention="blocked")
    seed_agent(env, A2)
    with Store.open(env.state_dir) as store, store.lease(f"agent:{A1}"):
        for argv in (("ack", A1), ("label", "add", A1, "a=1")):
            code, stdout, _ = run_inprocess(env, monkeypatch, capsys, *argv, "--json")
            body = json.loads(stdout)
            assert code == 12, stdout
            assert body["error"]["details"]["retryable"] is True
    assert env.agent(A1)["needs_attention"] == "blocked" and labels_of(env, A1) == []


def test_label_add_on_an_older_schema_exits_12_with_the_hint(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # R6b: the self-label store is opened without migrating
    seed_agent(env, A1, csession_id=SESSION)
    db = sqlite3.connect(env.state_dir / "state.db")
    with db:
        db.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    db.close()
    code, stdout, _ = run_inprocess(
        env, monkeypatch, capsys, "label", "add", "--session", SESSION, "pr=1", "--json",
        extra_env={"CSESSION_ID": SESSION, "CSESSION_DIR": str(env.state_dir)},
    )  # fmt: skip
    body = json.loads(stdout)
    assert code == 12 and "older schema" in body["error"]["message"]
    assert fakebin_untouched(env)


def fakebin_untouched(env: Env) -> bool:
    return env.fakebin.argvs("claude-session") == []


def hook_sink(env_: Env, fakebin: Fakebin, tmp_path: Path, cmd: str | None = None) -> Env:
    sink = tmp_path / "sink.jsonl"
    cmd = cmd or f"cat >> {sink}; echo >> {sink}"
    return Env(fakebin, env_.iso, f'hooks:\n  post-state-change:\n    - cmd: "{cmd}"\n')


def test_ack_delivers_its_event_with_the_stored_updated(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    # R6c and R9: the delivered payload carries the stored row's `updated`
    hooked = hook_sink(env, fakebin, tmp_path)
    seed_agent(env, A1, status="live", needs_attention="blocked")
    out(hooked.run("ack", A1, "--json"), ACK)
    sink = tmp_path / "sink.jsonl"
    hooked.wait_for(sink.exists, "the post-state-change hook")
    payload = json.loads(sink.read_text().splitlines()[0])
    schema_registry.validate(payload, "agent-spawn/hooks/post-state-change.json")
    from agent_spawn.agent_view import ts

    assert payload["agent"]["updated"] == ts(hooked.agent(A1)["updated"])
    assert payload["agent"]["needs_attention"] is None


def test_a_failing_hook_is_swallowed_by_ack(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    # R6c
    hooked = hook_sink(env, fakebin, tmp_path, cmd="exit 3")
    seed_agent(env, A1, needs_attention="blocked")
    body = out(hooked.run("ack", A1, "--json"), ACK)
    assert body["changed"] is True and hooked.agent(A1)["needs_attention"] is None


def test_label_edit_delivers_pending_outbox_events(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    # R12: the next mutating command delivers what an earlier command queued
    from agent_spawn.agent_view import agent_object
    from agent_spawn.hooks import agent_event

    hooked = hook_sink(env, fakebin, tmp_path)
    seed_agent(env, A1, status="live")
    with hooked.store() as store:
        row = store.get_agent(A1)
        assert row is not None
        store.add_events(
            A1,
            [
                agent_event(
                    "post-state-change",
                    agent_object(store, row),
                    reads_third_party=False,
                    extra={
                        "change": "agent",
                        "previous_status": "parked",
                        "previous_needs_attention": None,
                    },
                )
            ],
        )
    out(hooked.run("label", "add", A1, "a=1", "--json"), ADD)
    sink = tmp_path / "sink.jsonl"
    hooked.wait_for(sink.exists, "the queued event's hook")
    assert json.loads(sink.read_text().splitlines()[0])["agent"]["agent_id"] == A1


def test_label_dry_run_delivers_nothing(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    hooked = hook_sink(env, fakebin, tmp_path)
    seed_agent(env, A1)
    out(hooked.run("label", "add", A1, "a=1", "--dry-run", "--json"), ADD)
    assert not (tmp_path / "sink.jsonl").exists()


def test_human_label_output_prints_warnings_on_stderr(env: Env, fakebin: Fakebin) -> None:
    # R3
    fakebin.seed_session(SESSION)
    fakebin.script("claude-session", "label add", exit=12, error={"code": "failed", "message": "m"})
    seed_agent(
        env, A1, csession_id=SESSION, claude_session_path=str(fakebin.bin_dir / "claude-session")
    )
    done = env.run("label", "add", A1, "a=1")
    assert done.returncode == 0
    assert done.stderr.count("warning: ") == 1 and "queued for reconcile" in done.stderr
    assert "warning" not in done.stdout
    quiet = env.run("label", "add", A1, "b=2", "--json")
    assert "warning" not in quiet.stderr


def test_ack_log_warnings_are_reported_in_both_modes(env: Env) -> None:
    # R3: a log that cannot be written is a warning in the result and on stderr in human mode
    seed_agent(env, A1, needs_attention="blocked")
    seed_agent(env, A2, needs_attention="blocked")
    (env.state_dir / "log.jsonl").mkdir()
    body = out(env.run("ack", A1, "--json"), ACK)
    assert any("could not write the log" in w for w in body["agent"]["warnings"])
    done = env.run("ack", A2)
    assert done.returncode == 0
    assert "warning: could not write the log" in done.stderr
    assert "warning" not in done.stdout


def test_a_failed_resolution_closes_the_store(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # R8
    seed_agent(env, A1)
    closed: list[int] = []
    real = Store.close

    def counting(self: Store) -> None:
        closed.append(1)
        real(self)

    monkeypatch.setattr(Store, "close", counting)
    gone = "ag-zzzzzzzz"
    for argv in (("ack", gone), ("show", gone), ("label", "add", gone, "a=1")):
        closed.clear()
        code, _, _ = run_inprocess(env, monkeypatch, capsys, *argv, "--json")
        assert code == 10 and closed, argv
