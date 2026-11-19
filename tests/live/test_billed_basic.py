"""Billed scenarios, wave 1: real Claude Code (haiku, tiny prompts) through the whole chain.

Every test uses the ``billed_world`` fixture (credentials in the private home, the counting
``claude`` wrapper first on ``PATH``). Expected sessions and turns are in ``scenarios.py``.
These tests were run against the real tools on 2026-11-19, see docs/live-testing.md, Results.
Every failure message names the assumption it broke.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from .billed import (
    bsetup,
    cs_status,
    expect_parked,
    expected_park_after_hangup,
    launch_headless,
    launch_window,
    pid_gone,
    show,
    track_agent,
    wait_turn_done,
)
from .harness import World, check, wait_for

SC = pytest.mark.live_scenario


@SC("headless-launch-wait")
def test_headless_launch_wait_end_to_end(billed_world: World) -> None:
    """One real headless haiku turn through launch --wait: the result object, ``--result-file``
    (mode 0600, same JSON), the relay's recorded states from real hook events
    (``first_working_at``), agent-spawn's ``post-park`` hook payload, claude-session's status
    fields (taint, reads_third_party, csession_dir, csession_id, labels, session ids, turns, cost)
    and the labels on the session. Confirms #36 #9, #14, #15 and design.md D1, D43."""
    world = billed_world
    hook_out = world.root / "post-park.json"
    wt = bsetup(world, spawn_extra=f'hooks:\n  post-park:\n    - cmd: "cat > {hook_out}"\n')
    result_file = world.root / "out" / "result.json"
    result_file.parent.mkdir()
    done = launch_headless(
        world,
        wt,
        "Reply with the single word ok.",
        "--worktree",
        "none",
        "--cwd",
        str(wt.repo.clone),
        "--label",
        "ticket=LIVE-1",
        "--result-file",
        str(result_file),
        ok=(0,),
    )
    agent = done["agent"]
    aid = agent["agent_id"]
    check(
        done["outcome"] == "done" and done["exit_code"] == 0,
        "a clean headless run with a result and no report is an implicit done (implement kind)",
        "claude-session implicit_done / #36 #88 default",
        {k: done[k] for k in ("outcome", "end_reason", "exit_code")},
    )
    check(
        agent["status"] == "parked"
        and agent["needs_attention"] is None
        and agent["last_outcome"] == "done",
        "the agent parks done without attention",
        "interface.md Outcome table",
        {k: agent[k] for k in ("status", "needs_attention", "last_outcome")},
    )
    check(
        done["first_working_at"],
        "first_working_at is set (real hook events reached the relay)",
        "design.md D1/D42 / #36 #14",
        done["first_working_at"],
    )
    check(
        stat.S_IMODE(result_file.stat().st_mode) == 0o600,
        "the result file is mode 0600",
        "design.md D32",
    )
    check(
        json.loads(result_file.read_text())["agent"]["agent_id"] == aid,
        "the result file carries the same JSON",
        "design.md D32",
    )
    wait_for(hook_out.exists, 15, "the post-park hook to write its payload")
    payload = json.loads(hook_out.read_text())
    world.observe("agent-spawn.post-park.payload.keys", sorted(payload))
    check(
        "tainted" in payload
        and payload.get("agent_id", payload.get("agent", {}).get("agent_id")) == aid,
        "the post-park hook payload carries tainted and the agent",
        "design.md D9 / #36 #14 (hook payload fields)",
        sorted(payload),
    )
    status = cs_status(world, agent["csession_id"])
    run = status["run"]
    check(
        run["phase"] == "session" and status["claude_session_id"],
        "claude-session recorded the Claude session id (phase session)",
        "design.md D43 / #36 #15",
    )
    check(
        status["tainted"] is False and run["reads_third_party"] is False and status["csession_dir"],
        "status carries taint, reads_third_party and csession_dir",
        "#36 #15 self-labeling inputs",
    )
    check(
        {f"agent={aid}", "ticket=LIVE-1"} <= set(status["labels"]),
        "the session carries agent-spawn's labels (agent=<id>, user labels)",
        "design.md D12 / #36 #14",
        status["labels"],
    )
    check(
        run.get("num_turns") and run["num_turns"] >= 1,
        "the run took at least one turn",
        "claude-session status",
        run.get("num_turns"),
    )
    world.observe("claude-session.real-run.run.keys", sorted(run))
    world.observe("claude-session.real-run.cost_usd", run.get("cost_usd"))
    world.observe("claude-session.real-run.num_turns", run.get("num_turns"))
    listed = world.json_tool("claude-session", "list", "--label", f"agent={aid}")["sessions"]
    check(
        len(listed) == 1 and listed[0]["last_outcome"] == "done",
        "list --label finds the one session",
        "#36 #75 5",
    )
    shown = show(world, aid)
    check(shown["runs"][0]["first_working_at"], "show records first_working_at", "design.md D42")
    outcome_files = list(Path(status["csession_dir"]).rglob("*outcome*"))
    world.observe("claude-session.session-dir.outcome-files", [p.name for p in outcome_files])
    world.spawn_json("teardown", aid)


@SC("interactive-window-close")
def test_interactive_launch_in_a_real_window_and_window_close(billed_world: World) -> None:
    """A real interactive haiku session in a real tmux window through termctl: the trust dialog is
    pre-accepted, the relay sees working then idle from real hook events, the supervisor is the
    pane process (spike A8), the window carries the labels and the attention marker rules apply;
    closing the window parks the agent ``killed`` without attention and claude exits (D17).
    Confirms #36 #26, #27, design.md D13, D17, A8, B1."""
    world = billed_world
    wt = bsetup(world)
    launched = launch_window(world, wt)
    agent = launched["agent"]
    aid = agent["agent_id"]
    track_agent(world, aid)
    state = wait_turn_done(world, aid, timeout=180)
    check(
        state["session"]["state"] in ("idle", "reported"),
        "after the one-turn prompt the interactive session settles idle or reported (not "
        "needs-input); --no-auto-exit keeps it up",
        "design.md D1 / #36 #27 (real states from real hook events)",
        state["session"],
    )
    run = state["runs"][0]
    check(run["first_working_at"], "the relay recorded the first working event", "design.md D42")
    panes = world.json_tool("termctl", "list", "--label", f"agent={aid}")["panes"]
    check(len(panes) == 1, "one window labeled agent=<id>", "design.md D2")
    pane = panes[0]
    check(
        pane["pid"] == state["supervisor"]["pid"],
        "the pane process is the recorded supervisor (spike A8 with the real chain)",
        "#36 #26 / design.md A8",
        (pane["pid"], state["supervisor"]),
    )
    world.observe("agent-spawn.window.marker-after-idle", (pane["marker"], pane["marker_text"]))
    csid = agent["csession_id"]
    claude_pid = cs_status(world, csid)["run"].get("child_pid")
    check(
        claude_pid, "claude-session knows the claude pid", "FIDELITY.md wrapper_alive/child_alive"
    )
    world.track(state["supervisor"]["pid"], tree=True)
    killed = world.tmux("kill-window", "-t", pane["window_id"])
    check(killed.returncode == 0, "kill-window works", "design.md B3", killed.stderr)
    parked = expect_parked(world, aid, 90)
    check(
        parked["park_reason"] == expected_park_after_hangup(world, csid)
        and parked["needs_attention"] is None,
        "closing the window parks the agent killed (done when the session had already reported) "
        "without attention",
        "issue: window close parks killed / D13 D17",
        {k: parked[k] for k in ("park_reason", "needs_attention", "last_end_reason")},
    )
    wait_for(lambda: pid_gone(int(claude_pid)), 30, "claude to exit after the hangup")
    final = cs_status(world, csid)["run"]
    world.observe(
        "claude-session.window-close.final",
        {k: final.get(k) for k in ("outcome", "end_reason", "exit_code", "session_end_reason")},
    )
    world.spawn_json("teardown", aid)


@SC("report-resume-chain")
def test_no_implicit_done_run_and_the_automatic_report_resume_carry_the_flag(
    billed_world: World,
) -> None:
    """``launch --no-implicit-done``: a headless run with a result and no report ends
    ``exited-unreported`` (``no_report``); agent-spawn's automatic report resume (a second real
    run) inherits the override through claude-session's carry (same kind, no ``--kind``), so
    run 2 is not an implicit done either. Needs claude-session >= 0.43.0. Confirms #36 #88 and the
    report-resume chain of interface.md."""
    world = billed_world
    # kinds.implement.backstop defaults to true: the Stop hook would block once and make the agent
    # file a report. Switch it off so run 1 really ends without one (claude-session
    # interface.md, Kind profile fields; design.md "backstop: false").
    wt = bsetup(world, cs_extra="kinds:\n  implement:\n    backstop: false\n")
    done = launch_headless(
        world,
        wt,
        "Reply with the single word ok. Do not run claude-session and do not file any report "
        "or status of any kind.",
        "--no-implicit-done",
        "--worktree",
        "none",
        "--cwd",
        str(wt.repo.clone),
        ok=(0, 14, 16, 17, 18),
    )
    aid = done["agent"]["agent_id"]
    runs = show(world, aid)["runs"]

    def outcome_of(entry: dict) -> object:
        # a runs[] entry holds the run's outcome object under "outcome" (observed live), whose own
        # "outcome" is the string
        value = entry["outcome"]
        return value.get("outcome") if isinstance(value, dict) else value

    world.observe(
        "report-resume.runs", [(r["run"], r["auto_resumed"], outcome_of(r)) for r in runs]
    )
    if runs and outcome_of(runs[0]) not in (None, "exited-unreported"):
        world.observe("report-resume.run1-reported-anyway", outcome_of(runs[0]))
        world.spawn_json("teardown", aid, ok=(0, 20))
        pytest.skip(
            "report-resume-chain: the model filed a report in run 1 despite the prompt "
            f"(outcome {outcome_of(runs[0])}): the chain was not exercised"
        )
    check(
        len(runs) == 2 and runs[1]["auto_resumed"] is True,
        "run 1 ended unreported and agent-spawn started its automatic report resume (run 2)",
        "interface.md Supervision (automatic report resume) / #36 #88",
        [(r["run"], r["auto_resumed"], outcome_of(r)) for r in runs],
    )
    check(
        outcome_of(runs[0]) == "exited-unreported",
        "--no-implicit-done: a headless run with a result and no report ends exited-unreported",
        "#36 #88 (no_report)",
        outcome_of(runs[0]),
    )
    status = world.json_tool("claude-session", "status", done["agent"]["csession_id"], "--run", "2")
    check(
        status["run"]["implicit_done"] is False,
        "the automatic report resume inherits --no-implicit-done through claude-session's carry",
        "#36 #88 (carry through the automatic report resume)",
        status["run"]["implicit_done"],
    )
    final = show(world, aid)["agent"]
    world.observe(
        "report-resume.final",
        {k: final[k] for k in ("status", "park_reason", "last_outcome", "last_end_reason")},
    )
    world.spawn_json("teardown", aid, ok=(0, 20))
