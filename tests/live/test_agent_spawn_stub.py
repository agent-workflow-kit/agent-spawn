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
