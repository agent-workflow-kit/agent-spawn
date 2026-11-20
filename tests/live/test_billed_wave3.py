"""Billed scenarios, wave 3 (#100): the interactive cases the first billed runs could not reach.

See ``test_billed_basic.py`` for the conventions. Each test starts one real interactive haiku
session in a real tmux window. Expected sessions and turns are in ``scenarios.py``.
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import signal
import time

import pytest

from .billed import (
    bsetup,
    cs_status,
    expect_parked,
    expected_park,
    launch_window,
    pid_gone,
    show,
    signal_exact,
    track_agent,
    wait_agent,
    wait_turn_done,
)
from .harness import World, check, process_token, wait_for

SC = pytest.mark.live_scenario

#: A turn that outlasts the test: no tools, hundreds of short lines. The test closes the window
#: while it is writing.
BUSY_PROMPT = (
    "Do not use any tools. Write a numbered list from 1 to 1500, one number per line, "
    "each line as '<number>: <a different short sentence about the number>', and nothing else."
)


@SC("interactive-window-close-unreported")
def test_window_closed_mid_turn_before_any_report_parks_killed(billed_world: World) -> None:
    """The case the first billed runs could not reach (the real model files its report in a tiny
    first turn): one interactive session is kept busy by a long no-tools task, its window is
    closed while the turn is running and before any report exists. interface.md (Outcome table,
    ``killed`` with end reason ``forwarded_hup``; Signals: a closed window signals only the pane's
    leader, ``supervise``, which forwards to ``exec``): the agent parks ``killed`` without
    attention, claude exits, and claude-session has no report. The observed values are recorded
    whatever they are (#100)."""
    world = billed_world
    wt = bsetup(world)
    agent = launch_window(world, wt, BUSY_PROMPT)["agent"]
    aid, csid = agent["agent_id"], agent["csession_id"]
    state = track_agent(world, aid)
    # mid-turn: the relay saw the first working event and the session is still working
    state = wait_agent(
        world,
        aid,
        lambda s: (
            bool(s["runs"] and s["runs"][0].get("first_working_at"))
            and (s.get("session") or {}).get("state") == "working"
        ),
        "the session to be working on the long turn",
        180,
    )
    run = cs_status(world, csid)["run"]
    check(
        not run.get("report"),
        "no report exists before the window is closed (the turn is still running)",
        "scenario premise",
        run.get("report"),
    )
    claude_pid = int(run["child_pid"])
    pane = world.json_tool("termctl", "list", "--label", f"agent={aid}")["panes"][0]
    world.track(state["supervisor"]["pid"], tree=True)
    killed = world.tmux("kill-window", "-t", pane["window_id"])
    check(killed.returncode == 0, "kill-window works", "design.md B3", killed.stderr)
    parked = expect_parked(world, aid, 90)
    wait_for(lambda: pid_gone(claude_pid), 30, "claude to exit after the hangup")
    final = cs_status(world, csid)["run"]
    observed = {
        "agent": {k: parked.get(k) for k in ("park_reason", "needs_attention", "last_end_reason")},
        "run": {k: final.get(k) for k in ("outcome", "end_reason", "exit_code", "report")},
    }
    world.observe("window-close-unreported", observed)
    check(
        parked["park_reason"] == "killed" and parked["needs_attention"] is None,
        "an unreported interactive session killed at its window parks killed without attention",
        "interface.md Outcome table (killed, end reason forwarded_hup: parked, attention none, "
        "park reason killed)",
        observed,
    )
    check(
        not final.get("report"),
        "no report exists after the close (the model was still in its first turn)",
        "scenario premise / interface.md Outcome table",
        observed,
    )
    check(
        show(world, aid)["agent"]["last_outcome"] == "killed",
        "the recorded last outcome is killed",
        "interface.md Outcome table",
        show(world, aid)["agent"]["last_outcome"],
    )
    world.spawn_json("teardown", aid)


@SC("window-wrapper-dead-child-alive-real")
def test_exec_killed_under_a_real_interactive_claude_in_a_window(billed_world: World) -> None:
    """The wrapper-dead-child-alive row with a real interactive claude (#97, #100). SIGKILL of
    ``exec`` (the pid agent-spawn recorded, identity-checked) leaves claude without a parent-death
    signal (that is headless only); ``supervise``, the pane's process, ends when it sees exec die
    and the pane hangs up. What the real claude then does is recorded. interface.md (Liveness):
    a child still alive under a dead supervisor keeps the agent ``live`` with ``supervisor_lost``
    and ``needs_attention: lost``, and ``stop`` parks it ``killed``; a child that died makes the
    run ``lost`` and the agent parks ``lost``. The test fails only when the agent's state
    contradicts the observed child state."""
    world = billed_world
    wt = bsetup(world)
    agent = launch_window(world, wt)["agent"]
    aid, csid = agent["agent_id"], agent["csession_id"]
    track_agent(world, aid)
    state = wait_turn_done(world, aid, 180)
    run = wait_for(
        lambda: (
            (r := cs_status(world, csid)["run"]).get("child_alive") and r.get("wrapper_alive") and r
        ),
        30,
        "claude and its wrapper to be running",
    )
    wrapper, child = int(run["wrapper_pid"]), int(run["child_pid"])
    check(
        state["supervisor"].get("exec_pid") == wrapper,
        "the exec pid agent-spawn recorded is claude-session's wrapper pid",
        "design.md D19 / claude-session status run.wrapper_pid",
        (state["supervisor"], wrapper),
    )
    child_token = process_token(child)
    if child_token is not None:
        world.pids.append((child, child_token))
    world.track(state["supervisor"]["pid"], tree=True)
    signal_exact(world, wrapper, signal.SIGKILL, comm_contains="exec")
    wait_for(lambda: pid_gone(wrapper), 20, "the wrapper to be dead")
    settled = wait_agent(
        world,
        aid,
        lambda s: s["agent"]["status"] == "parked" or s["agent"].get("supervisor_lost") is True,
        "agent-spawn to notice the dead supervisor (parked, or live with supervisor_lost)",
        90,
    )
    # give a real claude time to react to the hangup of its terminal, then re-read
    deadline = time.monotonic() + 20
    while not pid_gone(child) and time.monotonic() < deadline:
        time.sleep(0.5)
    child_alive_first = not pid_gone(child) and process_token(child) == child_token
    world.spawn_json("reconcile", ok=(0, 12))
    after = show(world, aid)
    # The arm comes from claude-session's state read AFTER ``show``: a child that died between
    # the first look and the agent read must not be judged against the live row.
    cs_run = cs_status(world, csid)["run"]
    child_alive = bool(cs_run.get("child_alive"))
    if not child_alive and after["agent"]["status"] != "parked":
        # the child died after the agent state was read: let the agent notice, then re-read
        world.spawn_json("reconcile", ok=(0, 12))
        after = wait_agent(
            world, aid, lambda s: s["agent"]["status"] == "parked", "the agent to park", 60
        )
        cs_run = cs_status(world, csid)["run"]
    world.observe(
        "window-wrapper-dead-real",
        {
            "child_alive_after_wrapper_kill": child_alive_first,
            "child_alive_at_second_read": child_alive,
            "agent_at_first_notice": {
                k: settled["agent"].get(k) for k in ("status", "supervisor_lost", "needs_attention")
            },
            "agent_after_reconcile": {
                k: after["agent"].get(k)
                for k in ("status", "supervisor_lost", "needs_attention", "park_reason")
            },
            "run": {
                k: cs_run.get(k) for k in ("outcome", "end_reason", "wrapper_alive", "child_alive")
            },
        },
    )
    if child_alive:
        check(
            after["agent"]["status"] == "live"
            and after["agent"]["supervisor_lost"] is True
            and after["agent"]["needs_attention"] == "lost",
            "a real interactive claude outlived its wrapper: the agent stays live with "
            "supervisor_lost and attention lost",
            "interface.md Liveness (a still-live session under a dead supervisor)",
            after["agent"],
        )
        world.spawn_json("stop", aid, "--grace", "5s")
        parked = expect_parked(world, aid, 60)
        wait_for(lambda: pid_gone(child), 30, "claude to end after stop")
        check(
            parked["park_reason"] == "killed" and parked["supervisor_lost"] is False,
            "stop parks the orphaned window agent killed and clears supervisor_lost",
            "interface.md Outcome table / Liveness",
            {k: parked.get(k) for k in ("park_reason", "supervisor_lost")},
        )
    else:
        # The run is final. Its outcome is claude-session's record: ``done`` when the real model
        # had already filed its report in its first turn ("a reported run keeps its outcome",
        # end reason ``wrapper_lost``, observed 2026-11-20), ``lost`` when it had not. The agent
        # parks per the outcome table row of that outcome (and end reason, for ``killed``).
        outcome, end_reason = cs_run.get("outcome"), cs_run.get("end_reason")
        expected = expected_park(outcome, end_reason)
        check(
            expected is not None,
            "the dead real claude's run ended with an outcome and end reason that the outcome "
            "table has a row for",
            "interface.md Outcome table / claude-session Outcome",
            (outcome, end_reason),
        )
        agent = after["agent"]
        check(
            agent["status"] == "parked"
            and (agent["park_reason"], agent["needs_attention"]) == expected
            and agent["supervisor_lost"] is False,
            "the real interactive claude died with its terminal: the agent parks per the outcome "
            "table row of the run's recorded outcome and end reason (no supervisor_lost for a "
            "final run)",
            "interface.md Outcome table / Liveness",
            (outcome, end_reason, agent),
        )
    world.spawn_json("teardown", aid, ok=(0, 20))
