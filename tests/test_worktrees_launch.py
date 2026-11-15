"""``launch --worktree new|<path>`` and the records of a cwd against the fake binaries."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json
from tests.procguard import ProcTracker

REPO = "example.org/team/project"
LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait")


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def wtctl_calls(fakebin: Fakebin, command: str | None = None) -> list[list[str]]:
    calls = fakebin.argvs("wtctl")
    return [c for c in calls if command is None or c[0] == command]


def real_starts(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" not in a]


def records(fakebin: Fakebin) -> dict[str, Any]:
    path = Path(str(fakebin.log) + ".records.json")
    return json.loads(path.read_text()) if path.exists() else {}


def only_agent(env: Env) -> dict[str, Any]:
    agents = env.agents()
    assert len(agents) == 1, agents
    return agents[0]


def test_worktree_new_creates_holds_and_runs_in_the_worktree(env: Env, fakebin: Fakebin) -> None:
    done = env.run(*LAUNCH, "--repo", REPO, "--worktree", "new", "--branch", "feat/x",
                   "--base", "main")  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/launch.json")
    agent_id = body["agent"]["agent_id"]
    create = wtctl_calls(fakebin, "create")
    assert create == [
        [
            "create",
            f"--repo={REPO}",
            f"--holder={agent_id}",
            f"--name={agent_id}",
            "--branch=feat/x",
            "--base=main",
            f"--actor={env.agent(agent_id)['actor']}",
            "--json",
        ]
    ]
    record = next(iter(records(fakebin).values()))
    assert record["holders"] == [agent_id] and record["name"] == agent_id
    worktree = body["agent"]["worktree"]
    assert worktree == {"record_id": record["id"], "path": record["path"], "owned": True}
    assert body["agent"]["holds"] == [worktree]
    row = env.agent(agent_id)
    assert row["wt_record_id"] == record["id"] and row["wt_owned"] is True
    assert row["cwd"] == os.path.realpath(record["path"])
    argv = real_starts(fakebin)[-1]
    assert f"--cwd={os.path.realpath(record['path'])}" in argv and f"--repo={REPO}" in argv
    assert wtctl_calls(fakebin, "adopt") == []  # the creator already holds its own record
    assert wtctl_calls(fakebin, "list")  # the holds look the records up


def test_worktree_name_branch_detach_and_flags_reach_create(env: Env, fakebin: Fakebin) -> None:
    done = env.run(*LAUNCH, "--repo", REPO, "--worktree", "new", "--detach", "--ref", "a" * 40,
                   "--worktree-name", "my-tree", "--no-bootstrap", "--copy-env")  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    agent_id = result_json(done)["agent"]["agent_id"]
    create = wtctl_calls(fakebin, "create")[0]
    assert create[:4] == ["create", f"--repo={REPO}", f"--holder={agent_id}", "--name=my-tree"]
    assert "--detach" in create and f"--ref={'a' * 40}" in create
    assert "--no-bootstrap" in create and "--copy-env" in create
    assert env.agent(agent_id)["name"] == "my-tree"  # the window name defaults to the worktree name


def fake_throwaway(fakebin: Fakebin) -> dict[str, Any]:
    done = fakebin.run(
        "wtctl", "create", "--throwaway", "--repo", REPO, "--ref", "a" * 40, "--json"
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)["record"]  # type: ignore[no-any-return]


def fake_adopted(fakebin: Fakebin, path: Path, holder: str = "ag-zzzzzzzz") -> dict[str, Any]:
    done = fakebin.run("wtctl", "adopt", str(path), "--holder", holder, "--json")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)["record"]  # type: ignore[no-any-return]


def tree(tmp_path: Path, name: str = "tree") -> Path:
    path = tmp_path / name
    (path / "pkg").mkdir(parents=True)
    return path


# --- --worktree <path> ------------------------------------------------------------------------


def test_worktree_path_adopts_and_resolves_cwd_inside(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    root = tree(tmp_path)
    done = env.run(*LAUNCH, "--worktree", str(root), "--cwd", "pkg", "--repo", REPO)
    assert done.returncode == 0, (done.stdout, done.stderr)
    body = result_json(done)
    agent_id = body["agent"]["agent_id"]
    record = next(iter(records(fakebin).values()))
    assert wtctl_calls(fakebin, "adopt")[0] == [
        "adopt",
        str(root),
        f"--holder={agent_id}",
        f"--repo={REPO}",
        f"--actor={env.agent(agent_id)['actor']}",
        "--json",
    ]
    assert body["agent"]["worktree"] == {
        "record_id": record["id"],
        "path": record["path"],
        "owned": False,
    }
    assert env.agent(agent_id)["cwd"] == str(root.resolve() / "pkg")
    assert f"--cwd={root.resolve() / 'pkg'}" in real_starts(fakebin)[-1]
    assert env.agent(agent_id)["wt_owned"] is False


def test_worktree_path_through_a_symlink_runs_in_the_realpath(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    root = tree(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(root)
    done = env.run(*LAUNCH, "--worktree", str(link))
    assert done.returncode == 0, (done.stdout, done.stderr)
    agent_id = result_json(done)["agent"]["agent_id"]
    assert env.agent(agent_id)["cwd"] == str(root.resolve())


@pytest.mark.parametrize("cwd", ["../outside", "/etc", "pkg/../../outside", "link"])
def test_cwd_outside_the_adopted_worktree_exits_2_before_anything(
    env: Env, fakebin: Fakebin, tmp_path: Path, cwd: str
) -> None:
    root = tree(tmp_path)
    (tmp_path / "outside").mkdir()
    (root / "link").symlink_to(tmp_path / "outside")
    done = env.run(*LAUNCH, "--worktree", str(root), "--cwd", cwd)
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert env.agents() == [] and wtctl_calls(fakebin) == []


@pytest.mark.parametrize(
    "extra",
    [
        ("--branch", "-x"),
        ("--branch", "a b"),
        ("--branch", "a", "--detach"),
        ("--base", "--evil"),
        ("--ref", "abc"),
        ("--worktree-name", "-bad"),
        ("--worktree-name", "x" * 65),
        ("--cwd", "../up"),
        ("--cwd", "/abs"),
        ("--cwd", "session-dir"),
    ],
)
def test_worktree_new_usage_errors_exit_2_before_anything(
    env: Env, fakebin: Fakebin, extra: tuple[str, ...]
) -> None:
    done = env.run(*LAUNCH, "--worktree", "new", "--repo", REPO, *extra)
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert env.agents() == [] and wtctl_calls(fakebin) == []


def test_a_bad_repo_id_exits_2(env: Env, fakebin: Fakebin) -> None:
    done = env.run(*LAUNCH, "--worktree", "new", "--repo", "not a repo")
    assert done.returncode == 2 and env.agents() == [] and wtctl_calls(fakebin) == []


# --- the exit mapping -------------------------------------------------------------------------


def parked_row(env: Env) -> dict[str, Any]:
    row = only_agent(env)
    assert row["status"] == "parked", row["status"]
    return row


ESCALATED = {"error": {"code": "escalated", "message": "ask"}, "exit": 13}
FAILURES = [
    ({"error": {"code": "vetoed", "message": "no"}, "exit": 11}, 11, "vetoed", None, None, None),
    (ESCALATED, 13, "escalated", "escalated", "wtctl", None),
    (
        {"error": {"code": "busy", "message": "x"}, "exit": 12},
        12,
        "launch_failed",
        None,
        None,
        True,
    ),
    (
        {"error": {"code": "fetch_failed", "message": "x"}, "exit": 12},
        12,
        "launch_failed",
        "launch_failed",
        None,
        None,
    ),
    (
        {"error": {"code": "refused", "message": "x", "reason": "path_exists"}, "exit": 20},
        20,
        "launch_failed",
        "launch_failed",
        None,
        None,
    ),
    (
        {"error": {"code": "internal", "message": "x"}, "exit": 1},
        1,
        "launch_failed",
        "launch_failed",
        None,
        None,
    ),
]


@pytest.mark.parametrize(
    ("respond", "exit_code", "park", "attention", "escalated_by", "retryable"), FAILURES
)
def test_create_failures_park_per_the_spec(
    env: Env,
    fakebin: Fakebin,
    respond: dict[str, Any],
    exit_code: int,
    park: str,
    attention: str | None,
    escalated_by: str | None,
    retryable: bool | None,
) -> None:
    fakebin.script("wtctl", "create", **respond)
    done = env.run(*LAUNCH, "--repo", REPO, "--worktree", "new")
    assert done.returncode == exit_code, (done.stdout, done.stderr)
    error = json.loads(done.stdout)["error"]
    row = parked_row(env)
    assert error["agent_id"] == row["agent_id"] and error["tool"] == "wtctl"
    assert error["lower_error"]["code"] == respond["error"]["code"]
    assert (row["park_reason"], row["needs_attention"], row["escalated_by"]) == (
        park,
        attention,
        escalated_by,
    )
    assert (error["details"] or {}).get("retryable") == retryable
    assert row["csession_id"] is None and row["wt_record_id"] is None
    assert real_starts(fakebin) == []
    assert (env.state_dir / "prompts" / row["agent_id"]).is_dir()  # kept for the resume
    schema_registry.validate(json.loads(done.stdout), "agent-spawn/cli/error.json")


def test_bootstrap_failed_stores_the_record_and_keeps_the_hold(env: Env, fakebin: Fakebin) -> None:
    fakebin.script("wtctl", "create", bootstrap_failed="install")
    done = env.run(*LAUNCH, "--repo", REPO, "--worktree", "new")
    assert done.returncode == 12, (done.stdout, done.stderr)
    row = parked_row(env)
    record = next(iter(records(fakebin).values()))
    assert row["wt_record_id"] == record["id"] and row["wt_owned"] is True
    assert row["wt_path"] == record["path"]  # filled from wtctl show
    assert (row["park_reason"], row["needs_attention"]) == ("launch_failed", "launch_failed")
    with env.store() as store:
        assert [h["record_id"] for h in store.get_holds(row["agent_id"])] == [record["id"]]
    assert real_starts(fakebin) == []


def test_bootstrap_failed_still_stores_the_id_when_show_fails(env: Env, fakebin: Fakebin) -> None:
    fakebin.script("wtctl", "create", bootstrap_failed="install")
    fakebin.script("wtctl", "show", error={"code": "internal", "message": "x"}, exit=1)
    done = env.run(*LAUNCH, "--repo", REPO, "--worktree", "new")
    assert done.returncode == 12
    row = parked_row(env)
    record = next(iter(records(fakebin).values()))
    assert row["wt_record_id"] == record["id"] and row["wt_path"] is None


def test_adopt_failures_park_per_the_spec(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    root = tree(tmp_path)
    fakebin.script("wtctl", "adopt", error={"code": "busy", "message": "later"}, exit=12)
    done = env.run(*LAUNCH, "--worktree", str(root))
    assert done.returncode == 12
    row = parked_row(env)
    assert (row["park_reason"], row["needs_attention"]) == ("launch_failed", None)
    assert json.loads(done.stdout)["error"]["details"]["retryable"] is True


def test_adopt_other_failure_sets_attention(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    root = tree(tmp_path)
    fakebin.script("wtctl", "adopt", error={"code": "not_found", "message": "x"}, exit=10)
    done = env.run(*LAUNCH, "--worktree", str(root))
    assert done.returncode == 10
    row = parked_row(env)
    assert (row["park_reason"], row["needs_attention"]) == ("launch_failed", "launch_failed")


# --- the cwd rules over the records ---------------------------------------------------------------


def test_third_party_run_with_a_worktree_is_refused_before_allocation(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    for worktree in ("new", str(tree(tmp_path))):
        done = env.run(*LAUNCH, "--repo", REPO, "--worktree", worktree, "--reads-third-party")
        assert done.returncode == 20, (done.stdout, done.stderr)
        error = json.loads(done.stdout)["error"]
        assert error["reason"] == "third_party_cwd" and error["tool"] == "agent-spawn"
    assert env.agents() == [] and wtctl_calls(fakebin, "create") == []
    assert wtctl_calls(fakebin, "adopt") == []


def test_worktree_inside_a_throwaway_exits_2(env: Env, fakebin: Fakebin) -> None:
    throwaway = fake_throwaway(fakebin)
    done = env.run(*LAUNCH, "--worktree", throwaway["path"])
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert env.agents() == [] and wtctl_calls(fakebin, "adopt") == []


def test_trusted_run_in_a_throwaway_is_refused(env: Env, fakebin: Fakebin) -> None:
    throwaway = fake_throwaway(fakebin)
    sub = Path(throwaway["path"]) / "sub"
    sub.mkdir()
    for cwd in (throwaway["path"], str(sub)):
        done = env.run(*LAUNCH, "--cwd", cwd)
        assert done.returncode == 20, (done.stdout, done.stderr)
        error = json.loads(done.stdout)["error"]
        assert error["reason"] == "third_party_run" and error["tool"] == "agent-spawn"
    assert env.agents() == []


def test_third_party_run_at_a_throwaway_root_runs_and_is_not_held(
    env: Env, fakebin: Fakebin
) -> None:
    throwaway = fake_throwaway(fakebin)
    done = env.run(*LAUNCH, "--cwd", throwaway["path"], "--reads-third-party")
    assert done.returncode == 0, (done.stdout, done.stderr)
    body = result_json(done)
    assert body["agent"]["holds"] == [] and body["agent"]["worktree"] is None
    assert wtctl_calls(fakebin, "adopt") == []
    assert f"--cwd={os.path.realpath(throwaway['path'])}" in real_starts(fakebin)[-1]


def test_third_party_run_below_a_throwaway_root_is_refused(env: Env, fakebin: Fakebin) -> None:
    throwaway = fake_throwaway(fakebin)
    sub = Path(throwaway["path"]) / "sub"
    sub.mkdir()
    done = env.run(*LAUNCH, "--cwd", str(sub), "--reads-third-party")
    assert done.returncode == 20
    assert json.loads(done.stdout)["error"]["reason"] == "third_party_cwd"


def test_third_party_run_in_a_worktree_record_is_refused_even_at_a_throwaway_root(
    env: Env, fakebin: Fakebin
) -> None:
    throwaway = fake_throwaway(fakebin)
    around = {
        "id": "wt-bbbbbbbb",
        "path": str(Path(throwaway["path"]).parent),
        "object_type": "worktree",
        "state": "active",
        "name": "outer",
        "owned": False,
        "holders": [],
    }
    fakebin.script("wtctl", "list", stdout={"records": [throwaway, around], "foreign": []})
    done = env.run(*LAUNCH, "--cwd", throwaway["path"], "--reads-third-party")
    assert done.returncode == 20
    assert json.loads(done.stdout)["error"]["reason"] == "third_party_cwd"


def test_a_symlinked_cwd_into_a_throwaway_is_resolved_first(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    throwaway = fake_throwaway(fakebin)
    link = tmp_path / "innocent"
    link.symlink_to(throwaway["path"])
    done = env.run(*LAUNCH, "--cwd", str(link))
    assert done.returncode == 20
    assert json.loads(done.stdout)["error"]["reason"] == "third_party_run"


# --- holds ---------------------------------------------------------------------------------------


def test_worktree_none_with_a_cwd_inside_records_holds_every_record(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    outer = tree(tmp_path)
    inner = outer / "pkg"
    outer_record = fake_adopted(fakebin, outer)
    inner_record = fake_adopted(fakebin, inner)
    done = env.run(*LAUNCH, "--cwd", str(inner))
    assert done.returncode == 0, (done.stdout, done.stderr)
    body = result_json(done)
    agent_id = body["agent"]["agent_id"]
    held = {h["record_id"]: h for h in body["agent"]["holds"]}
    assert set(held) == {outer_record["id"], inner_record["id"]}
    assert all(h["owned"] is False for h in held.values())
    live = records(fakebin)
    assert agent_id in live[outer_record["id"]]["holders"]
    assert agent_id in live[inner_record["id"]]["holders"]
    assert body["agent"]["worktree"]["record_id"] == inner_record["id"]  # the nearest
    adopts = [a for a in wtctl_calls(fakebin, "adopt") if f"--holder={agent_id}" in a]
    assert sorted(a[1] for a in adopts) == sorted([str(outer.resolve()), str(inner.resolve())])


def test_hold_failures_map_like_adopt(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    root = tree(tmp_path)
    fake_adopted(fakebin, root)
    fakebin.script("wtctl", "adopt", error={"code": "busy", "message": "x"}, exit=12, times=1)
    done = env.run(*LAUNCH, "--cwd", str(root))
    assert done.returncode == 12
    row = parked_row(env)
    assert (row["park_reason"], row["needs_attention"]) == ("launch_failed", None)
    assert json.loads(done.stdout)["error"]["details"]["retryable"] is True
    assert real_starts(fakebin) == []


def test_a_failed_records_lookup_refuses_before_anything(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    fakebin.script("wtctl", "list", error={"code": "internal", "message": "x"}, exit=1)
    done = env.run(*LAUNCH, "--cwd", str(tmp_path))
    assert done.returncode == 1
    assert json.loads(done.stdout)["error"]["tool"] == "wtctl"
    assert env.agents() == []


# --- wtctl availability ---------------------------------------------------------------------------


def test_wtctl_missing_exits_12_when_a_lookup_or_worktree_needs_it(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    gone = tmp_path / "no" / "wtctl"
    missing = Env(fakebin, isolation_env, f"wtctl_cmd: ['{gone}']\n")
    for args in (("--repo", REPO, "--worktree", "new"), ("--cwd", str(tmp_path))):
        done = missing.run(*LAUNCH, *args)
        assert done.returncode == 12, (done.stdout, done.stderr)
        assert json.loads(done.stdout)["error"]["code"] == "tool_missing"
    assert missing.agents() == []


def test_wtctl_missing_does_not_matter_for_a_session_dir_run(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    missing = Env(fakebin, isolation_env, f"wtctl_cmd: ['{tmp_path / 'no' / 'wtctl'}']\n")
    done = missing.run(*LAUNCH, "--cwd", "session-dir")
    assert done.returncode == 0, (done.stdout, done.stderr)


def test_without_wtctl_a_trusted_run_runs_and_a_third_party_run_needs_session_dir(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    standalone = Env(fakebin, isolation_env, "wtctl_cmd: null\n")
    ok = standalone.run(*LAUNCH, "--cwd", str(tmp_path))
    assert ok.returncode == 0, (ok.stdout, ok.stderr)
    refused = standalone.run(*LAUNCH, "--cwd", str(tmp_path), "--reads-third-party")
    assert refused.returncode == 20
    assert json.loads(refused.stdout)["error"]["reason"] == "third_party_cwd"
    assert wtctl_calls(fakebin) == []
    flags = standalone.run(*LAUNCH, "--worktree", str(tmp_path))
    assert flags.returncode == 2


def test_the_wtctl_path_is_stored_with_the_agent(env: Env, fakebin: Fakebin) -> None:
    done = env.run(*LAUNCH, "--repo", REPO, "--worktree", "new")
    agent_id = result_json(done)["agent"]["agent_id"]
    assert env.agent(agent_id)["wtctl_path"] == str(fakebin.bin_dir / "wtctl")


# --- dry run ----------------------------------------------------------------------------------


def test_dry_run_plans_the_worktree_and_creates_nothing(env: Env, fakebin: Fakebin) -> None:
    done = env.run(
        "launch", "--mode", "headless", "--skill", "s", "--repo", REPO,
        "--worktree", "new", "--branch", "b", "--dry-run", "--json",
    )  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/launch-dry-run.json")
    assert body["worktree_plan"]["dry_run"] is True and body["worktree_plan"]["record"]["id"]
    assert body["tools"]["wtctl"] == [str(fakebin.bin_dir / "wtctl")]
    assert body["would_refuse"] is None
    assert records(fakebin) == {} and env.agents() == []
    assert "--dry-run" in wtctl_calls(fakebin, "create")[0]


def test_dry_run_reports_the_refusals_of_the_cwd_rules(env: Env, fakebin: Fakebin) -> None:
    throwaway = fake_throwaway(fakebin)
    base = ("launch", "--mode", "headless", "--skill", "s", "--dry-run", "--json")
    third = env.run(*base, "--repo", REPO, "--worktree", "new", "--reads-third-party")
    assert third.returncode == 20 and result_json(third)["would_refuse"] == "third_party_cwd"
    trusted = env.run(*base, "--cwd", throwaway["path"])
    assert trusted.returncode == 20 and result_json(trusted)["would_refuse"] == "third_party_run"
    fine = env.run(*base, "--cwd", throwaway["path"], "--reads-third-party")
    assert fine.returncode == 0 and result_json(fine)["would_refuse"] is None


# --- interruption -----------------------------------------------------------------------------


def test_a_signal_during_create_parks_killed_and_stores_no_record(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("wtctl", "create", hang=True, on_signal={"SIGTERM": {"exit": 143}})
    launcher = proc_tracker.popen(
        env.argv(*LAUNCH, "--repo", REPO, "--worktree", "new"),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    env.wait_for(lambda: bool(wtctl_calls(fakebin, "create")), "the in-flight create")
    create = next(i for i in fakebin.invocations("wtctl") if i.argv[0] == "create")
    fakebin.wait_for_hang(create.pid)
    launcher.send_signal(signal.SIGTERM)
    launcher.communicate(timeout=30)
    assert launcher.returncode == 143
    row = parked_row(env)
    assert (row["park_reason"], row["needs_attention"]) == ("killed", None)
    assert row["wt_record_id"] is None
    ended = next(i for i in fakebin.invocations("wtctl") if i.argv[0] == "create")
    assert signal.SIGTERM in ended.signals
