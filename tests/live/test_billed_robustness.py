"""Billed scenarios, wave 3: kills, signals, the minimal environment and jobs, real Claude Code.

See ``test_billed_basic.py`` for the conventions; these tests have not been run against a real
Claude Code yet.
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import json
import signal
import subprocess
from typing import Any

import pytest

from .billed import (
    PROMPT_OK,
    bsetup,
    cs_status,
    expect_parked,
    launch_window,
    pid_gone,
    show,
    signal_exact,
    spawn,
    wait_agent,
    wait_session_state,
)
from .harness import World, check, wait_for
from .test_wtctl import Wt

SC = pytest.mark.live_scenario

LONG_PROMPT = "Write the numbers from 1 to 1500, one per line, and nothing else."


def resume_headless(world: World, aid: str) -> dict[str, Any]:
    return spawn(
        world,
        "resume",
        aid,
        "--mode",
        "headless",
        "--model",
        "haiku",
        "--max-turns",
        "3",
        "--prompt",
        PROMPT_OK,
        "--wait",
        "--no-promote",
    )


def claude_pid(world: World, csession_id: str) -> int:
    pid = wait_for(
        lambda: cs_status(world, csession_id)["run"].get("child_pid"),
        60,
        "claude-session to record the child pid",
    )
    return int(pid)


@SC("resumable-after-kill-pane")
def test_claude_exits_when_its_pane_is_killed_and_the_session_resumes(billed_world: World) -> None:
    """Spike D1: ``tmux kill-pane`` on a window with a real interactive session. claude gets no
    signal of its own (the pane's process is hung up and forwards), exits, the agent parks killed
    and the session with a turn is resumable (a headless resume reaches ``done``). Confirms
    design.md D1, B1 and the issue's kill-pane check."""
    world = billed_world
    wt = bsetup(world)
    agent = launch_window(world, wt)["agent"]
    aid = agent["agent_id"]
    state = wait_session_state(world, aid, {"idle"}, 180)
    child = claude_pid(world, agent["csession_id"])
    pane = world.json_tool("termctl", "list", "--label", f"agent={aid}")["panes"][0]
    world.track(state["supervisor"]["pid"], tree=True)
    killed = world.tmux("kill-pane", "-t", pane["pane_id"])
    check(killed.returncode == 0, "kill-pane works", "design.md D1", killed.stderr)
    parked = expect_parked(world, aid, 90)
    check(
        parked["park_reason"] == "killed" and parked["needs_attention"] is None,
        "kill-pane parks the agent killed without attention",
        "design.md D1 / D13 (outcomes come from claude-session's recorded cause)",
        {k: parked[k] for k in ("park_reason", "needs_attention", "last_end_reason")},
    )
    wait_for(lambda: pid_gone(child), 30, "claude to exit")
    run = cs_status(world, agent["csession_id"])["run"]
    world.observe(
        "d1.kill-pane.run1",
        {k: run.get(k) for k in ("outcome", "end_reason", "exit_code", "session_end_reason")},
    )
    again = resume_headless(world, aid)
    check(
        again["agent"]["run"] == 2 and again["outcome"] == "done",
        "the session is resumable after kill-pane (run 2 reaches done)",
        "design.md D1 (every session with a turn is resumable)",
        {k: again[k] for k in ("outcome", "end_reason")},
    )
    world.spawn_json("teardown", aid)


def sigterm_case(world: World, wt: Wt, prompt: str, states: set[str]) -> None:
    agent = launch_window(world, wt, prompt)["agent"]
    aid = agent["agent_id"]
    state = wait_session_state(world, aid, states, 180)
    world.track(state["supervisor"]["pid"], tree=True)
    child = claude_pid(world, agent["csession_id"])
    signal_exact(world, child, signal.SIGTERM, comm_contains="claude")
    parked = expect_parked(world, aid, 120)
    run = cs_status(world, agent["csession_id"])["run"]
    world.observe(
        f"d2d3.{sorted(states)}.run1",
        {k: run.get(k) for k in ("outcome", "end_reason", "exit_code", "session_end_reason")},
    )
    world.observe(
        f"d2d3.{sorted(states)}.agent",
        {k: parked[k] for k in ("park_reason", "needs_attention", "last_end_reason")},
    )
    check(pid_gone(child), "SIGTERM ended claude", "design.md D2/D3 (SIGTERM: normal exit 143)")
    again = resume_headless(world, aid)
    check(
        again["agent"]["run"] == 2 and again["outcome"] == "done",
        "the session resumes after SIGTERM",
        "design.md D2/D3 (every session with a turn is resumable)",
        {k: again[k] for k in ("outcome", "end_reason")},
    )
    world.spawn_json("teardown", aid)


@SC("resumable-after-sigterm")
def test_sigterm_of_an_idle_claude_then_resume(billed_world: World) -> None:
    """Spike D2: SIGTERM (to the exact claude pid) of an idle interactive session: claude exits
    (143 in the spike), the run's outcome comes from claude-session's recorded cause, and the
    session resumes. Confirms design.md D2."""
    wt = bsetup(billed_world)
    sigterm_case(billed_world, wt, PROMPT_OK, {"idle"})


@SC("resumable-after-sigterm")
def test_sigterm_of_a_claude_in_the_middle_of_a_turn_then_resume(billed_world: World) -> None:
    """Spike D3: SIGTERM of a claude that is generating a long answer; same checks as D2.
    Confirms design.md D3."""
    wt = bsetup(billed_world)
    sigterm_case(billed_world, wt, LONG_PROMPT, {"working"})


@SC("minimal-environment")
def test_claude_authenticates_with_only_path_and_the_private_home(billed_world: World) -> None:
    """Spike D4: ``claude -p`` with an environment of ``PATH`` and the private ``HOME`` only (plus
    the one credential variable, when credentials are variables) authenticates and answers. Runs
    through the counting wrapper. Confirms design.md D4 (the handoff's full environment is more
    than enough)."""
    world = billed_world
    env = {"PATH": f"{world.bin}:/usr/bin:/bin", "HOME": str(world.home)}
    if world.auth is not None:
        env.update(world.auth.env)
    done = subprocess.run(
        [
            world.claude_path,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "json",
            "--max-turns",
            "1",
            "--",
            PROMPT_OK,
        ],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=world.timeout(180),
        check=False,
        cwd=world.root,
    )
    check(
        done.returncode == 0,
        "claude -p with only PATH and HOME exits 0",
        "design.md D4",
        done.stderr[-300:],
    )
    result = json.loads(done.stdout)
    check(
        result.get("is_error") is False and result.get("result"),
        "claude authenticated and answered",
        "design.md D4 (claude authenticated with only PATH set)",
        {k: result.get(k) for k in ("is_error", "subtype")},
    )
    world.observe("d4.minimal-env.keys", sorted(result))


def long_headless(world: World, wt: Wt, *extra: str) -> dict[str, Any]:
    return spawn(
        world,
        "launch",
        "--mode",
        "headless",
        "--prompt",
        LONG_PROMPT,
        "--model",
        "haiku",
        "--max-turns",
        "3",
        "--timeout",
        "4m",
        "--worktree",
        "none",
        "--cwd",
        str(wt.repo.clone),
        *extra,
    )


@SC("liveness-kills")
def test_killing_the_supervisor_only_and_exec_only(billed_world: World) -> None:
    """#36 #75 3 and 4 with real headless runs (the detached supervisor is the parent of ``exec``):
    SIGKILL of the supervisor only leaves ``exec`` and claude alive (claude-session still reports
    the run live) and agent-spawn sets ``supervisor_lost`` once; SIGKILL of ``exec`` only gives
    ``wrapper_alive: false`` / ``child_alive: true`` and ``supervisor_lost`` without finalizing
    the run. ``stop`` then ends each run; a late ``claude-session reconcile`` and an
    ``agent-spawn reconcile`` do not deliver ``post-park`` twice. Signals go to exact PIDs
    recorded by agent-spawn and claude-session, after an identity check."""
    world = billed_world
    wt = bsetup(world)
    hook_log = world.root / "post-park.log"
    for label, victim in (("supervisor", "supervisor"), ("exec", "wrapper")):
        launched = long_headless(
            world, wt, "--hook", f"post-park=cat >> {hook_log}; echo >> {hook_log}"
        )
        aid = launched["agent"]["agent_id"]
        csid = launched["agent"]["csession_id"]
        state = wait_agent(
            world,
            aid,
            lambda s: s["agent"]["status"] == "live" and s["supervisor"]["alive"],
            "a live run",
            120,
        )
        wait_for(
            lambda csid=csid: cs_status(world, csid)["run"].get("child_alive"), 60, "claude to run"
        )
        run = cs_status(world, csid)["run"]
        target = state["supervisor"]["pid"] if victim == "supervisor" else run["wrapper_pid"]
        comm = "supervise" if victim == "supervisor" else "exec"
        signal_exact(world, int(target), signal.SIGKILL, comm_contains=comm)
        lost = wait_agent(
            world,
            aid,
            lambda s: s["agent"]["supervisor_lost"],
            f"supervisor_lost after killing {label}",
            60,
        )
        run_after = cs_status(world, csid)["run"]
        world.observe(
            f"liveness.{label}.status",
            {k: run_after.get(k) for k in ("outcome", "wrapper_alive", "child_alive")},
        )
        check(
            run_after["outcome"] is None,
            f"killing {label} only does not finalize the run",
            "#36 #75 3",
            run_after.get("outcome"),
        )
        if victim == "wrapper":
            check(
                run_after["wrapper_alive"] is False and run_after["child_alive"] is True,
                "SIGKILL of exec only: wrapper_alive false, child_alive true",
                "#36 #75 3 / FIDELITY.md wrapper_alive",
                (run_after["wrapper_alive"], run_after["child_alive"]),
            )
        else:
            check(
                run_after["child_alive"] is True,
                "SIGKILL of the supervisor only: claude lives on",
                "design.md B9 / #36 #75 3",
            )
        check(
            lost["agent"]["supervisor_lost"] is True,
            "agent-spawn sets supervisor_lost",
            "#36 #75 3",
        )
        spawn(world, "stop", aid, ok=(0, 12))
        wait_for(
            lambda csid=csid: cs_status(world, csid)["run"]["outcome"] is not None,
            60,
            "the stopped run to be final",
        )
        parked = expect_parked(world, aid, 60)
        world.observe(
            f"liveness.{label}.parked",
            {k: parked[k] for k in ("park_reason", "needs_attention", "supervisor_lost")},
        )
        before = hook_log.read_text() if hook_log.exists() else ""
        world.tool("claude-session", "reconcile", "--json", auth=True)
        world.spawn_json("reconcile", ok=(0, 12))
        after = hook_log.read_text() if hook_log.exists() else ""
        check(
            before == after,
            "a late reconcile delivers no second post-park (no hook runs twice)",
            "#36 #75 4 (late delivery after a liveness park)",
            (len(before), len(after)),
        )
        again = show(world, aid)["agent"]
        check(
            again["last_outcome"] == parked["last_outcome"],
            "a late delivery does not record a second last_outcome",
            "#36 #75 4",
            (again["last_outcome"], parked["last_outcome"]),
        )
        world.spawn_json("teardown", aid, ok=(0, 20))


@SC("jobsctl-cancel-force")
def test_jobsctl_cancel_force_of_a_job_running_launch_wait(billed_world: World) -> None:
    """A jobsctl job whose command is ``agent-spawn launch --wait`` with a long headless prompt;
    ``jobsctl cancel --force`` SIGTERMs the waiter, which forwards (D44) and keeps waiting until the
    agent parked ``killed``; no claude is left running. Confirms the issue's jobsctl scenario and
    design.md D20, D44 (a waiter SIGKILLed after its grace would leave the agent running)."""
    world = billed_world
    wt = bsetup(world)
    job_env = {
        "PATH": f"{world.bin}:/usr/bin:/bin",
        "HOME": str(world.home),
        "XDG_STATE_HOME": str(world.state),
        "XDG_CONFIG_HOME": str(world.config_home),
        "TMUX_TMPDIR": str(world.tmuxdir),
    }
    argv = [
        world.agent_spawn_path,
        "launch",
        "--mode",
        "headless",
        "--prompt",
        LONG_PROMPT,
        "--model",
        "haiku",
        "--max-turns",
        "3",
        "--timeout",
        "4m",
        "--worktree",
        "none",
        "--cwd",
        str(wt.repo.clone),
        "--wait",
        "--json",
    ]
    env_args: list[str] = []
    for name, value in job_env.items():
        env_args += ["--env", f"{name}={value}"]
    pass_env = [a for name in (world.auth.env if world.auth else {}) for a in ("--pass-env", name)]
    added = world.json_tool(
        "jobsctl",
        "add",
        "--kill-grace",
        "60s",
        *env_args,
        *pass_env,
        "--cwd",
        str(wt.repo.clone),
        "--",
        *argv,
        auth=True,
    )
    job = added.get("job", added)
    job_id = job["id"]
    world.observe("jobsctl.add", {k: job.get(k) for k in ("id", "status")})
    agent = wait_for(
        lambda: next(iter(world.spawn_json("list", ok=(0,))["agents"]), None),
        120,
        "the job to launch an agent",
    )
    aid = agent["agent_id"]
    wait_agent(
        world,
        aid,
        lambda s: (
            s["agent"]["status"] == "live" and (s.get("session") or {}).get("state") == "working"
        ),
        "the run to work",
        120,
    )
    cancelled = world.tool(
        "jobsctl",
        "--json",
        "cancel",
        "--force",
        "--wait-timeout",
        "90s",
        "--actor",
        "human:live",
        job_id,
        auth=True,
        timeout=150,
    )
    world.observe("jobsctl.cancel", (cancelled.returncode, cancelled.stdout[-300:]))
    shown = world.json_tool("jobsctl", "show", job_id)
    world.observe(
        "jobsctl.show", {k: shown.get(k, shown.get("job", {}).get(k)) for k in ("status", "reason")}
    )
    parked = expect_parked(world, aid, 90)
    check(
        parked["park_reason"] == "killed" and parked["needs_attention"] is None,
        "cancel --force through launch --wait ends the run and parks the agent killed without attention",
        "design.md D44 / D20",
        {k: parked[k] for k in ("park_reason", "needs_attention", "last_end_reason")},
    )
    run = cs_status(world, parked["csession_id"])["run"]
    check(
        run["outcome"] is not None,
        "the run is final (no claude left running)",
        "design.md D20 residual risk",
        run.get("outcome"),
    )
    world.spawn_json("teardown", aid, ok=(0, 20))
