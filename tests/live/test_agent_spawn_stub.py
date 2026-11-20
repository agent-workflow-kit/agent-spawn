"""agent-spawn against the real claude-session, wtctl, termctl and tmux, with a stub ``claude``.

No Claude Code runs: ``claude`` is a shell stub (``World.install_claude(stub_body=...)``) that
claude-session executes like the real binary. claude-session decides ``phase: session`` when
its child is running, so the whole agent-spawn lifecycle (launch, handoff, supervision, relay
bookkeeping that does not need Claude hooks, resume, fork, holds, teardown, liveness, dead
panes) runs against the real tools. What a real Claude adds (hook events, reports, resumable
conversations) is checked by the billed scenarios.
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import pytest

from .billed import pid_gone
from .harness import World, check, process_token, wait_for
from .test_wtctl import Wt

SC = pytest.mark.live_scenario


def stub_setup(world: World, stub: str, *, agent_spawn_config: str = "") -> Wt:
    world.install_claude(stub_body=stub)
    world.claude_session_config("lost_grace: 120s\n")
    world.write_config(
        "agent-spawn", "auto_report_resume: false\nhandoff_timeout: 30s\n" + agent_spawn_config
    )
    wt = Wt(world)
    world.trust(wt.repo.clone)
    return wt


def agent_of(done: dict[str, Any]) -> dict[str, Any]:
    return done["agent"]


def show(world: World, agent_id: str) -> dict[str, Any]:
    return world.spawn_json("show", agent_id)


def wait_status(world: World, agent_id: str, status: str, timeout: float = 90) -> dict[str, Any]:
    box: dict[str, Any] = {}

    def done() -> bool:
        box["show"] = show(world, agent_id)
        return box["show"]["agent"]["status"] == status

    wait_for(done, timeout, f"agent {agent_id} to be {status}", interval=0.5)
    return box["show"]


@SC("as-stub-lifecycle")
def test_headless_launch_with_a_new_worktree_resume_fork_and_teardown(world: World) -> None:
    """#36 #20-#23 (holds, fork holds, ``held`` passed through, teardown order, ``forget`` of
    holders) and #75 5 (``list --label agent=<id>`` after resume and after ``--fork``): a headless
    stub run in a ``--worktree new`` worktree, a second run (resume), an in-place fork, then the
    original's teardown refused while the fork holds the record, and both torn down."""
    wt = stub_setup(world, "exit 0")
    launched = world.spawn_json(
        "launch",
        "--mode",
        "headless",
        "--prompt",
        "reply with the single word ok",
        "--model",
        "haiku",
        "--repo",
        wt.repo.repo_id,
        "--worktree",
        "new",
        "--wait",
        ok=(0, 14),
    )
    agent = agent_of(launched)
    aid = agent["agent_id"]
    record = agent["worktree"]
    world.observe("agent-spawn.launch.worktree", record)
    records = wt.ok("list")["records"]
    check(
        len(records) == 1 and aid in records[0]["holders"] and records[0]["name"] == aid,
        "launch --worktree new creates a wtctl record named after the agent id and held by it",
        "design.md D6/D23 (name = agent id, holder = agent id)",
        records,
    )
    cwd = Path(records[0]["path"])
    check(
        agent["cwd"] == str(cwd),
        "the agent runs in the worktree root",
        "design.md D11",
        agent["cwd"],
    )
    sessions = world.json_tool("claude-session", "list", "--label", f"agent={aid}")["sessions"]
    check(
        len(sessions) == 1,
        "the session carries the agent=<id> label (found exactly once)",
        "#36 #75 5 / #14",
        sessions,
    )
    status = world.json_tool("claude-session", "status", agent["csession_id"])
    check(
        f"agent={aid}" in status["labels"] and status["tainted"] is False,
        "claude-session status carries the labels and taint agent-spawn compares",
        "#36 #15 (status fields taint/reads_third_party/csession_dir)",
        status["labels"],
    )
    check(
        status["csession_dir"] and status["run"]["reads_third_party"] is False,
        "status carries csession_dir and run.reads_third_party",
        "#36 #15",
    )
    shown = show(world, aid)
    world.observe("agent-spawn.show.keys", sorted(shown))

    # resume: a second run of the same session; the label search still finds one session
    resumed = world.spawn_json(
        "resume", aid, "--mode", "headless", "--prompt", "reply ok", "--wait", ok=(0, 14)
    )
    check(
        resumed["agent"]["run"] == 2,
        "resume makes run 2",
        "design.md Resume order",
        resumed["agent"]["run"],
    )
    check(
        len(world.json_tool("claude-session", "list", "--label", f"agent={aid}")["sessions"]) == 1,
        "after a resume `list --label agent=<id>` lists the session once",
        "#36 #75 5",
    )

    # in-place fork: a new agent holding the same worktree record
    forked = world.spawn_json(
        "resume", aid, "--fork", "--mode", "headless", "--prompt", "reply ok", "--wait", ok=(0, 14)
    )
    fork = forked["agent"]
    fid = fork["agent_id"]
    check(
        fid != aid and fork["forked_from"] == aid,
        "a fork is a new agent with forked_from",
        "design.md D25/D46",
    )
    after_fork = world.json_tool("claude-session", "list", "--label", f"agent={aid}")["sessions"]
    world.observe(
        "claude-session.fork.label-search-of-source", [s["csession_id"] for s in after_fork]
    )
    check(
        [s["csession_id"] for s in after_fork] == [agent["csession_id"]],
        "after resume --fork `list --label agent=<source id>` returns exactly the source session "
        "(a fork must not inherit the source's agent label)",
        "#36 #75 5 / FIDELITY.md (fork label inheritance)",
        [s["labels"] for s in after_fork],
    )
    holders = wt.ok("show", records[0]["id"])["holders"]
    check(
        holders == sorted([aid, fid]),
        "the fork holds the same worktree record; holders come back sorted",
        "#36 #21/#22 (fork holds, sorted holders)",
        holders,
    )
    refused = world.agent_spawn("teardown", aid, "--json")
    err = json.loads(refused.stdout)["error"]
    world.observe("agent-spawn.teardown.held", err)
    check(
        refused.returncode == 20 and "held" in json.dumps(err),
        "the original's teardown is refused while the fork holds the worktree (wtctl held, exit 20)",
        "#36 #22/#23 (held passed through by teardown with lower_error)",
        (refused.returncode, refused.stdout[-300:]),
    )
    check(cwd.exists(), "a refused teardown leaves the worktree", "design.md D14")
    world.spawn_json("teardown", fid)
    check(
        wt.ok("show", records[0]["id"])["holders"] == [aid],
        "tearing the fork down releases its hold (forget --holder)",
        "#36 #22 release_fork_holds",
    )
    torn = world.spawn_json("teardown", aid)
    world.observe("agent-spawn.teardown.keys", sorted(torn))
    gone = wt.ok("show", records[0]["id"])
    check(
        gone["state"] == "removed" and not cwd.exists(),
        "teardown removes the owned worktree through wtctl remove --yes --holder",
        "#36 #23 (real teardown)",
        gone["state"],
    )
    branches = world.git("branch", "--list", "wt/*", cwd=wt.repo.clone)
    world.observe("git.branches.after-teardown", branches)
    for session in (agent["csession_id"], fork["csession_id"]):
        unknown = world.tool("claude-session", "status", session, "--json")
        check(
            unknown.returncode == 10,
            "teardown forgets the session (its id is unknown afterwards)",
            "design.md D14 (forget the session)",
            unknown.returncode,
        )
    tomb = world.agent_spawn("show", aid, "--json")
    tomb_agent = json.loads(tomb.stdout)["agent"] if tomb.returncode == 0 else {}
    check(
        tomb.returncode == 0
        and tomb_agent.get("status") == "torn_down"
        and tomb_agent.get("torn_down_at"),
        "a full id of a torn-down agent shows its tombstone (status torn_down, torn_down_at set)",
        "interface.md show (a full id of a torn-down agent shows its tombstone)",
        (tomb.returncode, tomb.stdout[-200:]),
    )


PRIME_WTCTL = """#!/bin/sh
# Runs the real wtctl, but before agent-spawn's real `create` it makes the same holder a record on
# the branch under another name, so wtctl answers agent-spawn's create with that record (reused).
real={real}
case " $* " in *" --dry-run "*) exec "$real" "$@" ;; esac
if [ "$1" = create ]; then
  for arg in "$@"; do
    case "$arg" in --holder=*) holder="${{arg#--holder=}}" ;; esac
  done
  "$real" create --repo={repo} --holder="$holder" --name=older --branch=feat/x --json >/dev/null \
    || exit 99
fi
exec "$real" "$@"
"""


@SC("wtctl-reuse-mismatch")
def test_a_reused_record_under_another_name_is_refused_and_left_alone(world: World) -> None:
    """#99: the same holder with a new ``--name`` and an explicit ``--branch`` an earlier record
    holds gets that record back from the real wtctl (``reused: true``, the old name): agent-spawn
    refuses it (``launch_failed`` / ``worktree_mismatch``, exit 12), parks the agent, starts no
    session and leaves the record as it was. wtctl is the real one, wrapped only to create the
    earlier record for the holder agent-spawn allocated."""
    wrapper = world.root / "wtctl-prime"
    # the wrapper needs the repo id, known once the Wt (its repo) exists
    wt = stub_setup(world, "exit 0")
    real = world.tools["wtctl"]
    assert real is not None
    wrapper.write_text(PRIME_WTCTL.format(real=real.path, repo=wt.repo.repo_id))
    wrapper.chmod(0o700)
    world.write_config(
        "agent-spawn",
        f"auto_report_resume: false\nhandoff_timeout: 30s\nwtctl_cmd: [{json.dumps(str(wrapper))}]\n",
    )
    failed = world.spawn_json(
        "launch",
        "--mode",
        "headless",
        "--prompt",
        "reply with the single word ok",
        "--model",
        "haiku",
        "--repo",
        wt.repo.repo_id,
        "--worktree",
        "new",
        "--branch",
        "feat/x",
        "--wait",
        ok=(12,),
    )
    error = failed["error"]
    aid = error["agent_id"]
    world.observe("agent-spawn.launch.reuse-mismatch", error)
    records = wt.ok("list")["records"]
    check(
        len(records) == 1 and records[0]["name"] == "older" and records[0]["holders"] == [aid],
        "the real wtctl answered the create with the holder's earlier record (nothing new was made)",
        "wtctl interface.md Create order, Retry after a crash (the retry's other flags are not compared); "
        "docs/live-testing.md First unbilled run",
        records,
    )
    check(
        error["code"] == "launch_failed"
        and error["tool"] == "agent-spawn"
        and error["details"]["cause"] == "worktree_mismatch"
        and records[0]["id"] in error["message"],
        "the launch is refused naming the returned record, with details.cause worktree_mismatch",
        "interface.md Launch order step 4 (a reused record that is not the planned worktree)",
        error,
    )
    shown = show(world, aid)["agent"]
    check(
        shown["status"] == "parked"
        and shown["park_reason"] == "launch_failed"
        and shown["worktree"] is None,
        "the agent is parked launch_failed and recorded no worktree",
        "interface.md Launch order step 4",
        shown,
    )
    check(
        world.json_tool("claude-session", "list", "--label", f"agent={aid}")["sessions"] == [],
        "no session was started",
        "interface.md Launch order step 4 (nothing launched)",
    )
    check(
        records[0]["state"] == "active" and Path(records[0]["path"]).exists(),
        "the reused record and its worktree are untouched (never adopted, forgotten or removed)",
        "interface.md Launch order step 4",
        records[0],
    )


@SC("as-stub-lifecycle")
def test_window_launch_pane_pid_binding_and_window_close(world: World) -> None:
    """Spike A8 with the real termctl and its hooks (#36 #26: ``pid`` equals the window process,
    which ``exec``s ``supervise``; the peer check matched) and spike B1/B2 through the real
    chain (#36 #27): closing the window parks the agent ``killed`` without attention, and the
    stub child gets hung up through ``supervise`` and ``exec`` (D17)."""
    wt = stub_setup(world, "while :; do sleep 0.2; done")
    launched = world.spawn_json(
        "launch",
        "--mode",
        "interactive",
        "--prompt",
        "reply ok",
        "--model",
        "haiku",
        "--worktree",
        "none",
        "--cwd",
        str(wt.repo.clone),
        timeout=120,
    )
    agent = agent_of(launched)
    aid = agent["agent_id"]
    window = agent["window"]
    panes = world.json_tool("termctl", "list", "--label", f"agent={aid}")["panes"]
    check(
        len(panes) == 1 and panes[0]["id"] == window["term_id"],
        "the window is labeled agent=<id>",
        "design.md D2",
    )
    pane = panes[0]
    tmux_pid = int(
        world.tmux("display-message", "-p", "-t", pane["pane_id"], "#{pane_pid}").stdout.strip()
    )
    state = wait_status(world, aid, "live")
    supervisor = state["supervisor"]
    check(
        pane["pid"] == tmux_pid == supervisor["pid"],
        "termctl's pid, tmux's pane_pid and the recorded supervisor are one process (spike A8; the "
        "window process execs supervise)",
        "#36 #26 / design.md A8 D19 D21",
        (pane["pid"], tmux_pid, supervisor["pid"]),
    )
    check(
        Path(f"/proc/{tmux_pid}/cmdline").read_bytes().count(b"supervise") >= 1,
        "the pane process is supervise",
        "design.md D19",
    )
    check(
        "TMUX"
        in {
            e.split(b"=")[0].decode()
            for e in Path(f"/proc/{tmux_pid}/environ").read_bytes().split(b"\0")
            if e
        },
        "terminal variables come from the pane",
        "design.md A5 D4",
    )
    exec_pid = supervisor["exec_pid"]
    wait_for(lambda: exec_pid and os.path.exists(f"/proc/{exec_pid}"), 15, "exec to run")
    world.track(tmux_pid, tree=True)
    world.observe("agent-spawn.window.labels", pane["labels"])
    # close the window: SIGHUP reaches supervise only; it forwards (D17); the agent parks killed
    killed = world.tmux("kill-window", "-t", pane["window_id"])
    check(killed.returncode == 0, "kill-window", "design.md B3", killed.stderr)
    parked = wait_status(world, aid, "parked", 60)["agent"]
    world.observe(
        "agent-spawn.window-close.park",
        {
            k: parked[k]
            for k in ("park_reason", "needs_attention", "last_outcome", "last_end_reason")
        },
    )
    check(
        parked["park_reason"] == "killed" and parked["needs_attention"] is None,
        "closing the window parks the agent killed without attention",
        "issue: window close (parked killed) / design.md D13 D17",
        {k: parked[k] for k in ("park_reason", "needs_attention", "last_end_reason")},
    )
    wait_for(
        lambda: (
            not Path(f"/proc/{exec_pid}").exists()
            or Path(f"/proc/{exec_pid}/stat").read_text().rsplit(")", 1)[-1].split()[0] == "Z"
        ),
        20,
        "exec to end after the hangup",
    )


@SC("as-dead-pane-reconcile")
def test_a_killed_supervisor_leaves_a_dead_pane_that_reconcile_closes(world: World) -> None:
    """#36 #79 (``--keep-on-exit failed`` leaves a dead pane for a failed exit; teardown and
    ``reconcile`` close it on a real server; ``list --label agent=<id> --dead`` finds it on the
    default server) and #75 3 (SIGKILL of the supervisor only: ``supervisor_lost``); design.md
    D51: reconcile closes a dead pane of a parked agent without attention only with
    ``reconcile.close_dead_panes_after``."""
    wt = stub_setup(
        world,
        "while :; do sleep 0.2; done",
        agent_spawn_config="reconcile:\n  close_dead_panes_after: 1s\n",
    )
    launched = world.spawn_json(
        "launch",
        "--mode",
        "interactive",
        "--prompt",
        "reply ok",
        "--model",
        "haiku",
        "--worktree",
        "none",
        "--cwd",
        str(wt.repo.clone),
        timeout=120,
    )
    aid = agent_of(launched)["agent_id"]
    state = wait_status(world, aid, "live")
    sup = state["supervisor"]["pid"]
    token = process_token(sup)
    world.pids.append((sup, token or ""))
    world.track(state["supervisor"]["exec_pid"], tree=True)
    check(token is not None, "the supervisor is a live process", "#36 #75 3")
    pane = world.json_tool("termctl", "list", "--label", f"agent={aid}")["panes"][0]
    os.kill(
        sup, signal.SIGKILL
    )  # the exact PID recorded by agent-spawn and checked against the pane
    dead = wait_for(
        lambda: world.json_tool("termctl", "list", "--label", f"agent={aid}", "--dead")["panes"],
        20,
        "a dead pane after the supervisor was killed",
    )
    check(
        dead[0]["exit_signal"] == 9 and dead[0]["id"] == pane["id"],
        "a SIGKILLed pane process leaves a dead pane (exit_signal 9) under --keep-on-exit failed",
        "#36 #27 / #79 (failed exit leaves a dead pane)",
        dead[0],
    )
    time.sleep(2)  # past close_dead_panes_after
    after = world.spawn_json("show", aid)
    world.observe(
        "agent-spawn.supervisor-killed.show",
        {
            k: after["agent"][k]
            for k in ("status", "park_reason", "needs_attention", "supervisor_lost")
        },
    )
    reconciled = world.spawn_json("reconcile", ok=(0, 12))
    world.observe("agent-spawn.reconcile.output", reconciled)
    # The first pass itself changes the agent (it parks it and delivers the held events), which
    # restarts the idle clock (``updated``): a dead pane is then ``too_new``. Let the clock age
    # past close_dead_panes_after and run a second pass.
    time.sleep(2)
    reconciled = world.spawn_json("reconcile", ok=(0, 12))
    world.observe("agent-spawn.reconcile.second-pass", reconciled)
    remaining = world.json_tool("termctl", "list", "--label", f"agent={aid}", "--dead")["panes"]
    parked = show(world, aid)["agent"]
    if parked["needs_attention"] is None:
        check(
            not remaining,
            "reconcile closes the dead pane of a parked agent without attention once "
            "close_dead_panes_after has passed",
            "#36 #79 end-to-end reconcile closing a pane on a real tmux server / D51",
            (parked["needs_attention"], remaining),
        )
    else:
        # an unacknowledged attention item keeps the pane (D51); acknowledge and run again
        world.spawn_json("ack", aid)
        time.sleep(1.5)
        world.spawn_json("reconcile", ok=(0, 12))
        check(
            not world.json_tool("termctl", "list", "--label", f"agent={aid}", "--dead")["panes"],
            "after ack, reconcile closes the dead pane",
            "design.md D51 (an unacknowledged needs_attention always keeps it)",
        )
    done = world.spawn_json("teardown", aid)
    check(done is not None, "teardown after the pane was closed succeeds", "design.md D14 step 5")


def _argv_of(pid: int) -> list[str] | None:
    """The argv of ``pid`` as separate elements, or ``None`` when it is gone."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return [a.decode(errors="replace") for a in raw.split(b"\0") if a]


def _term_count(path: Path) -> int:
    return len(path.read_text().splitlines()) if path.exists() else 0


def _run_status(world: World, csid: str) -> dict[str, Any]:
    return world.json_tool("claude-session", "status", csid)["run"]


@SC("as-wrapper-dead-child-alive")
def test_a_wrapper_killed_under_a_sigterm_ignoring_child_is_supervisor_lost_until_stop(
    world: World,
) -> None:
    """#75 3 and 4, the row the billed ``liveness-kills`` cannot reach (issue #97): a headless
    ``claude`` normally dies with a SIGKILLed ``exec`` (claude-session gives it
    ``PR_SET_PDEATHSIG(SIGTERM)``, so the run ends ``lost``). A child that ignores SIGTERM
    survives: the stub ``trap '' TERM``. claude-session then reports wrapper dead, child alive and
    no outcome; agent-spawn keeps the agent ``live`` with ``supervisor_lost`` (and
    ``needs_attention: lost``, set once on the transition), ``reconcile`` changes nothing a second
    time, and ``stop`` ends the orphan (SIGTERM ignored, then SIGKILL after the grace) and parks
    it ``killed``. Signals go to the exact PIDs claude-session and agent-spawn recorded."""
    hook_log = world.root / "post-park.log"
    ready = world.root / "stub-ready"
    term_log = world.root / "stub-term.log"
    # the stub ignores SIGTERM but records each one, and signals once the trap is installed
    wt = stub_setup(
        world,
        f"trap 'echo term >> {term_log}' TERM; touch {ready}; while :; do sleep 0.2; done",
        agent_spawn_config=f'hooks:\n  post-park:\n    - cmd: "cat > /dev/null; echo ran >> {hook_log}"\n',
    )
    launched = world.spawn_json(
        "launch",
        "--mode",
        "headless",
        "--prompt",
        "reply with the single word ok",
        "--model",
        "haiku",
        "--worktree",
        "none",
        "--cwd",
        str(wt.repo.clone),
        timeout=120,
    )
    aid = agent_of(launched)["agent_id"]
    csid = agent_of(launched)["csession_id"]
    world.note_session(csid)
    state = wait_status(world, aid, "live")
    world.track(int(state["supervisor"]["pid"]), tree=True)
    run = wait_for(
        lambda: (r := _run_status(world, csid)).get("child_alive") and r.get("wrapper_alive") and r,
        30,
        "the stub child and its wrapper to be running",
    )
    wrapper, child = int(run["wrapper_pid"]), int(run["child_pid"])
    record_token = process_token(wrapper)
    child_token = process_token(child)
    if child_token is not None:
        world.pids.append((child, child_token))  # cleanup tracks it before any check can fail
    check(
        child_token is not None and wrapper != child,
        "the wrapper and the stub child are two live processes",
        "claude-session interface.md status run fields (wrapper_pid, child_pid)",
        (wrapper, child),
    )
    wait_for(ready.exists, 20, "the stub to install its SIGTERM trap")
    check(
        state["supervisor"].get("exec_pid") == wrapper,
        "the exec pid agent-spawn recorded is claude-session's wrapper pid",
        "design.md D19 / claude-session status run.wrapper_pid",
        (state["supervisor"].get("exec_pid"), wrapper),
    )
    wrapper_token = process_token(wrapper)
    argv = _argv_of(wrapper)
    check(
        argv is not None
        and wrapper_token == record_token
        and any(Path(a).name == "claude-session" for a in argv)
        and "exec" in argv,
        "the wrapper is still the process claude-session recorded and runs claude-session exec",
        "#36 #75 3",
        (argv, record_token, wrapper_token),
    )
    os.kill(wrapper, signal.SIGKILL)  # exact PID recorded by claude-session, identity checked above
    wait_for(
        lambda: (
            process_token(wrapper) != wrapper_token
            or Path(f"/proc/{wrapper}/stat").read_text().rsplit(")", 1)[-1].split()[0] == "Z"
        ),
        20,
        "the wrapper to be dead",
    )
    after = wait_for(
        lambda: (r := _run_status(world, csid)).get("wrapper_alive") is False and r,
        30,
        "claude-session to see the wrapper dead",
    )
    # PR_SET_PDEATHSIG sent the stub one SIGTERM when the wrapper died; bash defers the trap
    # until its foreground ``sleep`` exits, so the log line can lag claude-session's view
    wait_for(lambda: _term_count(term_log) >= 1, 20, "the stub to log the PDEATHSIG SIGTERM")
    terms_at_death = _term_count(term_log)
    check(
        terms_at_death == 1,
        "the wrapper's death sent the stub exactly one SIGTERM (PR_SET_PDEATHSIG)",
        "claude-session interface.md Signals / #97",
        terms_at_death,
    )
    world.observe("wrapper-dead.terms", terms_at_death)
    world.observe(
        "wrapper-dead.status",
        {k: after.get(k) for k in ("outcome", "end_reason", "wrapper_alive", "child_alive")},
    )
    check(
        after["child_alive"] is True and process_token(child) == child_token,
        "a SIGTERM-ignoring child survives the death of its wrapper (PR_SET_PDEATHSIG is ignored)",
        "claude-session interface.md Signals / #97",
        (after.get("child_alive"), process_token(child)),
    )
    check(
        after["outcome"] is None,
        "wrapper dead, child alive: claude-session records no outcome",
        "claude-session interface.md status; #36 #75 3",
        after.get("outcome"),
    )
    lost = wait_status_flag(world, aid)
    world.observe("wrapper-dead.supervisor", lost.get("supervisor"))
    check(
        (lost.get("supervisor") or {}).get("alive") is False,
        "the supervisor ended on its own (exec died with no outcome to read)",
        "interface.md Liveness (a dead supervisor) / supervise.py end_of_run",
        lost.get("supervisor"),
    )
    world.observe(
        "wrapper-dead.agent",
        {k: lost["agent"][k] for k in ("status", "needs_attention", "supervisor_lost")},
    )
    check(
        lost["agent"]["status"] == "live"
        and lost["agent"]["supervisor_lost"] is True
        and lost["agent"]["needs_attention"] == "lost",
        "agent-spawn keeps the agent live with supervisor_lost and needs_attention lost",
        "interface.md Liveness (a still-live session under a dead supervisor) / #36 #75 3",
        {k: lost["agent"][k] for k in ("status", "needs_attention", "supervisor_lost")},
    )
    listed = [a for a in world.spawn_json("list")["agents"] if a["agent_id"] == aid]
    check(
        len(listed) == 1 and listed[0]["supervisor_lost"] is True and listed[0]["status"] == "live",
        "list reports the same",
        "interface.md Liveness / list",
        listed,
    )
    changes = world.spawn_json("reconcile", ok=(0, 12))
    check(
        not [c for c in changes["changes"] if c["agent_id"] == aid and c["kind"] == "parked"],
        "reconcile does not park an orphan that is still live",
        "interface.md Liveness (stop is the remedy)",
        changes["changes"],
    )
    log = (world.state / "agent-spawn" / "log.jsonl").read_text().splitlines()
    flagged = [
        r
        for r in map(json.loads, log)
        if r.get("agent_id") == aid
        and r.get("event") == "reconcile_change"
        and (r.get("details") or {}).get("kind") == "supervisor_lost"
    ]
    check(
        len(flagged) == 1,
        "supervisor_lost is logged once however many times liveness runs",
        "interface.md Liveness (needs_attention only on the transition) / #36 #75 3",
        len(flagged),
    )
    check(
        process_token(child) == child_token and _term_count(term_log) == terms_at_death,
        "no liveness check signalled the orphan: the stub received no SIGTERM after its "
        "wrapper died",
        "interface.md Liveness (liveness never stops the run)",
        (_term_count(term_log), terms_at_death),
    )
    # the remedy: stop ends the orphan (SIGTERM is ignored, so SIGKILL after the grace)
    stopped = world.spawn_json("stop", aid, "--grace", "2s")
    world.observe("wrapper-dead.stop", stopped)
    session_stop = stopped["session_stop"]
    check(
        stopped["action"] == "session_stop"
        and session_stop["ended"] is True
        and session_stop["run_exit_code"] == 15
        and session_stop["outcome"] == "killed"
        and session_stop["end_reason"] == "stopped",
        "stop exits 0 and claude-session's stop reports the run ended killed (run_exit_code 15)",
        "interface.md stop (session_stop is claude-session's stop result) / outcome table",
        {k: session_stop.get(k) for k in ("ended", "run_exit_code", "outcome", "end_reason")},
    )
    check(
        [(s["pid"], s["role"], s["signal"]) for s in session_stop["signals"]]
        == [(child, "child", "SIGTERM"), (child, "child", "SIGKILL")],
        "stop signalled only the orphaned child, SIGTERM first, then SIGKILL after the grace",
        "claude-session interface.md stop (signals)",
        session_stop["signals"],
    )
    wait_for(lambda: pid_gone(child), 30, "the stub child to be gone")
    check(
        _term_count(term_log) == terms_at_death + 1,
        "the stub received exactly one more SIGTERM, from stop, and survived it until SIGKILL",
        "claude-session interface.md stop (SIGTERM, then SIGKILL after the grace)",
        (_term_count(term_log), terms_at_death),
    )
    final = _run_status(world, csid)
    check(
        final["outcome"] is not None and final["child_alive"] is False,
        "after stop the run is final and the child gone",
        "interface.md stop / claude-session stop",
        {k: final.get(k) for k in ("outcome", "end_reason", "child_alive")},
    )
    parked = wait_status(world, aid, "parked", 60)["agent"]
    world.observe(
        "wrapper-dead.parked",
        {
            k: parked[k]
            for k in ("park_reason", "needs_attention", "supervisor_lost", "last_outcome")
        },
    )
    check(
        parked["park_reason"] == "killed" and parked["supervisor_lost"] is False,
        "stop parks the orphaned agent killed and clears supervisor_lost",
        "interface.md outcome table / Liveness (supervisor_lost clears when the run is final)",
        {k: parked[k] for k in ("park_reason", "needs_attention", "supervisor_lost")},
    )

    def hook_runs() -> int:
        return hook_log.read_text().count("ran\n") if hook_log.exists() else 0

    wait_for(lambda: hook_runs() >= 1, 15, "post-park to run for the stop's park")
    world.spawn_json("reconcile", ok=(0, 12))
    check(
        hook_runs() == 1,
        "post-park ran exactly once; a late reconcile delivers no second one",
        "#36 #75 4 / interface.md post-park (once per park)",
        hook_runs(),
    )
    world.spawn_json("teardown", aid, ok=(0, 20))


def wait_status_flag(world: World, agent_id: str, timeout: float = 60) -> dict[str, Any]:
    box: dict[str, Any] = {}

    def lost() -> bool:
        box["show"] = show(world, agent_id)
        return box["show"]["agent"]["supervisor_lost"] is True

    wait_for(lost, timeout, f"agent {agent_id} to be supervisor_lost", interval=0.5)
    return box["show"]


@SC("as-wrapper-dead-child-alive-window")
def test_a_wrapper_killed_under_a_signal_ignoring_child_in_a_window(world: World) -> None:
    """#97 follow-up (#100): the interactive-window variant of the wrapper-dead-child-alive row.
    An interactive ``claude`` has no parent-death signal, and the stub ignores SIGTERM and SIGHUP,
    so SIGKILL of ``exec`` (the pid agent-spawn recorded) leaves the child running in the window.
    Everything else follows the headless row: the run has no outcome, the supervisor ends (and
    its window process with it: a dead pane), the agent stays ``live`` with ``supervisor_lost``,
    and ``stop`` parks it ``killed``."""
    ready = world.root / "stub-ready"
    sig_log = world.root / "stub-sig.log"
    wt = stub_setup(
        world,
        f"trap 'echo term >> {sig_log}' TERM; trap 'echo hup >> {sig_log}' HUP; touch {ready}; while :; do sleep 0.2; done",
    )
    launched = world.spawn_json(
        "launch",
        "--mode",
        "interactive",
        "--prompt",
        "reply ok",
        "--model",
        "haiku",
        "--worktree",
        "none",
        "--cwd",
        str(wt.repo.clone),
        timeout=120,
    )
    agent = agent_of(launched)
    aid, csid = agent["agent_id"], agent["csession_id"]
    world.note_session(csid)
    state = wait_status(world, aid, "live")
    world.track(int(state["supervisor"]["pid"]), tree=True)
    run = wait_for(
        lambda: (r := _run_status(world, csid)).get("child_alive") and r.get("wrapper_alive") and r,
        30,
        "the stub child and its wrapper to be running",
    )
    wrapper, child = int(run["wrapper_pid"]), int(run["child_pid"])
    wait_for(ready.exists, 20, "the stub to install its traps")
    child_token = process_token(child)
    if child_token is not None:
        world.pids.append((child, child_token))
    wrapper_token = process_token(wrapper)
    argv = _argv_of(wrapper)
    world.observe(
        "window-wrapper-dead.before", (state["supervisor"], run["wrapper_pid"], run["child_pid"])
    )
    check(
        argv is not None and "exec" in argv and state["supervisor"].get("exec_pid") == wrapper,
        "the exec pid agent-spawn recorded is the live claude-session exec wrapper",
        "design.md D19 / claude-session status run.wrapper_pid",
        (argv, state["supervisor"]),
    )
    os.kill(wrapper, signal.SIGKILL)
    wait_for(
        lambda: (
            process_token(wrapper) != wrapper_token
            or Path(f"/proc/{wrapper}/stat").read_text().rsplit(")", 1)[-1].split()[0] == "Z"
        ),
        20,
        "wrapper dead",
    )
    after = wait_for(
        lambda: (r := _run_status(world, csid)).get("wrapper_alive") is False and r,
        30,
        "claude-session to see the wrapper dead",
    )
    check(
        after["child_alive"] is True
        and process_token(child) == child_token
        and after["outcome"] is None,
        "wrapper dead, child alive, no outcome: an interactive child has no parent-death signal and the stub ignores SIGHUP/SIGTERM",
        "claude-session interface.md Signals (PR_SET_PDEATHSIG is headless only) / #97",
        {k: after.get(k) for k in ("outcome", "child_alive", "wrapper_alive")},
    )
    lost = wait_status_flag(world, aid)
    world.observe(
        "window-wrapper-dead.agent",
        {k: lost["agent"][k] for k in ("status", "needs_attention", "supervisor_lost")},
    )
    check(
        lost["agent"]["status"] == "live"
        and lost["agent"]["supervisor_lost"] is True
        and lost["agent"]["needs_attention"] == "lost"
        and (lost["supervisor"] or {}).get("alive") is False,
        "the supervisor ended on its own; agent-spawn keeps the agent live with supervisor_lost and attention lost (the window variant of the headless row)",
        "interface.md Liveness (a still-live session under a dead supervisor) / #36 #75 3 / #97",
        (
            lost["agent"]["status"],
            lost["agent"]["supervisor_lost"],
            lost["agent"]["needs_attention"],
            lost["supervisor"],
        ),
    )
    panes = world.json_tool("termctl", "list", "--label", f"agent={aid}", "--dead")["panes"]
    check(
        len(panes) == 1 and panes[0]["state"] == "dead" and panes[0]["exit_status"] == 1,
        "the window's pane process (supervise) ended: the pane is dead (--keep-on-exit failed, exit status 1) while the child lives on",
        "design.md D51 / #36 #27 (keep-on-exit failed)",
        panes,
    )
    hups = sig_log.read_text().split() if sig_log.exists() else []
    world.observe("window-wrapper-dead.signals", hups)
    check(
        hups.count("term") == 0 and hups.count("hup") <= 1,
        "no liveness check signalled the orphan: at most the single SIGHUP of the dying pane, never a SIGTERM",
        "interface.md Liveness (liveness never stops the run)",
        hups,
    )
    changes = world.spawn_json("reconcile", ok=(0, 12))
    check(
        not [c for c in changes["changes"] if c["agent_id"] == aid and c["kind"] == "parked"]
        and process_token(child) == child_token,
        "reconcile does not park the live orphan nor signal it",
        "interface.md Liveness (stop is the remedy)",
        changes["changes"],
    )
    stopped = world.spawn_json("stop", aid, "--grace", "2s")
    session_stop = stopped["session_stop"]
    check(
        session_stop["ended"] is True
        and session_stop["outcome"] == "killed"
        and [(s["pid"], s["signal"]) for s in session_stop["signals"]]
        == [(child, "SIGTERM"), (child, "SIGKILL")],
        "stop ends the orphan (SIGTERM ignored, SIGKILL after the grace) and the run is killed",
        "claude-session interface.md stop (signals)",
        {k: session_stop.get(k) for k in ("ended", "outcome", "signals")},
    )
    wait_for(lambda: pid_gone(child), 30, "the stub child to be gone")
    parked = wait_status(world, aid, "parked", 60)["agent"]
    check(
        parked["park_reason"] == "killed" and parked["supervisor_lost"] is False,
        "stop parks the orphaned window agent killed and clears supervisor_lost",
        "interface.md outcome table / Liveness",
        {k: parked[k] for k in ("park_reason", "needs_attention", "supervisor_lost")},
    )
    world.spawn_json("teardown", aid, ok=(0, 20))
    check(
        not world.json_tool("termctl", "list", "--label", f"agent={aid}", "--dead")["panes"],
        "teardown closes the dead window",
        "design.md D14 step 5 / D51",
    )
