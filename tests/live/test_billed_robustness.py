"""Billed scenarios, wave 3: kills, signals, the minimal environment and jobs, real Claude Code.

See ``test_billed_basic.py`` for the conventions; these tests have not been run against a real
Claude Code yet.
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from .billed import (
    MAX_TURNS,
    PROMPT_OK,
    bsetup,
    cs_status,
    expect_parked,
    expected_park_after_hangup,
    launch_window,
    pid_gone,
    show,
    signal_exact,
    spawn,
    track_agent,
    wait_agent,
    wait_session_state,
    wait_turn_done,
)
from .harness import LiveError, World, check, wait_for, write_executable
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
        MAX_TURNS,
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
    track_agent(world, aid)
    state = wait_turn_done(world, aid, 180)  # the run has had a turn: it is resumable
    child = claude_pid(world, agent["csession_id"])
    pane = world.json_tool("termctl", "list", "--label", f"agent={aid}")["panes"][0]
    world.track(state["supervisor"]["pid"], tree=True)
    killed = world.tmux("kill-pane", "-t", pane["pane_id"])
    check(killed.returncode == 0, "kill-pane works", "design.md D1", killed.stderr)
    parked = expect_parked(world, aid, 90)
    check(
        parked["park_reason"] == expected_park_after_hangup(world, agent["csession_id"], parked)
        and parked["needs_attention"] is None,
        "kill-pane parks the agent killed (done when the session had already reported) without "
        "attention",
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
    track_agent(world, aid)
    if states == {"idle"}:
        state = wait_turn_done(world, aid, 180)  # a turn happened: the session is resumable
    else:
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
        MAX_TURNS,
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
    hook_log = world.root / "post-park.log"
    wt = bsetup(
        world,
        spawn_extra=f'hooks:\n  post-park:\n    - cmd: "cat > /dev/null; echo ran >> {hook_log}"\n',
    )
    for label, victim in (("supervisor", "supervisor"), ("exec", "wrapper")):
        launched = long_headless(world, wt)
        aid = launched["agent"]["agent_id"]
        csid = launched["agent"]["csession_id"]
        state = track_agent(world, aid)
        wait_for(
            lambda csid=csid: cs_status(world, csid)["run"].get("child_alive"), 60, "claude to run"
        )
        run = cs_status(world, csid)["run"]
        target = state["supervisor"]["pid"] if victim == "supervisor" else run["wrapper_pid"]
        comm = "supervise" if victim == "supervisor" else "exec"
        signal_exact(world, int(target), signal.SIGKILL, comm_contains=comm)
        try:
            lost = wait_agent(
                world,
                aid,
                lambda s: s["agent"]["supervisor_lost"],
                f"supervisor_lost after killing {label}",
                60,
            )
        except LiveError as err:
            now = show(world, aid)["agent"]
            raise LiveError(
                f"{err}; agent now: "
                f"{ {k: now.get(k) for k in ('status', 'park_reason', 'needs_attention', 'supervisor_lost', 'last_end_reason')} }; "
                f"claude-session run: "
                f"{ {k: cs_status(world, csid)['run'].get(k) for k in ('outcome', 'end_reason', 'wrapper_alive', 'child_alive', 'state')} }"
            ) from err
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
        ran = f"post-park hook runs after the {label} case"

        def hook_runs() -> int:
            return hook_log.read_text().count("ran\n") if hook_log.exists() else 0

        expected = 1 if label == "supervisor" else 2
        wait_for(lambda expected=expected: hook_runs() >= expected, 15, ran)
        before = hook_runs()
        check(
            before == expected,
            "post-park ran exactly once for this park (before the late reconcile)",
            "#36 #75 4 / interface.md post-park (once per park)",
            before,
        )
        world.tool("claude-session", "reconcile", "--json")
        world.spawn_json("reconcile", ok=(0, 12))
        after = hook_runs()
        check(
            after == before,
            "a late reconcile delivers no second post-park (no hook runs twice)",
            "#36 #75 4 (late delivery after the stopped run's park)",
            (before, after),
        )
        again = show(world, aid)["agent"]
        check(
            again["last_outcome"] == parked["last_outcome"],
            "a late delivery does not record a second last_outcome",
            "#36 #75 4",
            (again["last_outcome"], parked["last_outcome"]),
        )
        world.spawn_json("teardown", aid, ok=(0, 20))


def write_job_wrapper(world: World) -> Path:
    """A 0700 script in the private root that sources the credential variables from a 0600 file in
    the private root (nothing is stored in jobsctl's state or argv) and execs its arguments.

    jobsctl's ``--pass-env`` reads only from its ``env_file``, which jobsctl accepts only when
    every ancestor directory is writable by the owner alone; the private root is under ``/tmp``
    (1777), so that file would always be ``env_file_unsafe`` there (src-jobsctl env_file.py,
    ``_dir_trusted``). In login-copy mode ``HOME`` suffices and the file is empty."""
    auth_file = world.root / "job-auth.env"
    lines = [
        f"export {name}={shlex.quote(value)}\n"
        for name, value in (world.auth.env if world.auth else {}).items()
    ]
    fd = os.open(auth_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write("".join(lines))
    wrapper = world.root / "job-run.sh"
    write_executable(wrapper, f'#!/bin/sh\n. {shlex.quote(str(auth_file))}\nexec "$@"\n')
    return wrapper


@SC("jobsctl-cancel-force")
def test_jobsctl_cancel_force_of_a_job_running_launch_wait(billed_world: World) -> None:
    """A jobsctl job whose command is ``agent-spawn launch --wait`` with a long headless prompt;
    ``jobsctl cancel --force`` SIGTERMs the waiter, which forwards (D44) and keeps waiting until the
    agent parked ``killed``; cancel exits 0, the job ends ``canceled`` (``forced``) and no claude is
    left running. Confirms the issue's jobsctl scenario and design.md D20, D44 (a waiter SIGKILLed
    after its grace would leave the agent running)."""
    world = billed_world
    wt = bsetup(world)
    wrapper = write_job_wrapper(world)
    job_env = {
        "PATH": f"{world.bin}:/usr/bin:/bin",
        "HOME": str(world.home),
        "XDG_STATE_HOME": str(world.state),
        "XDG_CONFIG_HOME": str(world.config_home),
        "TMUX_TMPDIR": str(world.tmuxdir),
    }
    argv = [
        str(wrapper),
        world.agent_spawn_path,
        "launch",
        "--mode",
        "headless",
        "--prompt",
        LONG_PROMPT,
        "--model",
        "haiku",
        "--max-turns",
        MAX_TURNS,
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
    added = world.json_tool(
        "jobsctl",
        "add",
        "--kill-grace",
        "60s",
        *env_args,
        "--cwd",
        str(wt.repo.clone),
        "--",
        *argv,
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
    state = track_agent(world, aid)
    wait_agent(
        world,
        aid,
        lambda s: (
            s["agent"]["status"] == "live" and (s.get("session") or {}).get("state") == "working"
        ),
        "the run to work",
        120,
    )
    child = claude_pid(world, state["agent"]["csession_id"])
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
        timeout=150,
    )
    world.observe("jobsctl.cancel", (cancelled.returncode, cancelled.stdout[-300:]))
    check(
        cancelled.returncode == 0,
        "cancel --force of the running job exits 0 once the job is terminal",
        "jobsctl interface.md cancel (exit 12 cancel_pending on a timeout)",
        (cancelled.returncode, cancelled.stdout[-200:], cancelled.stderr[-200:]),
    )
    shown = world.json_tool("jobsctl", "show", job_id)
    final = shown.get("job", shown)
    check(
        final.get("status") == "canceled" and final.get("reason") == "forced",
        "the job ends canceled with reason forced",
        "jobsctl interface.md cancel / status table (canceled, forced)",
        {k: final.get(k) for k in ("status", "reason")},
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
    wait_for(lambda: pid_gone(child), 30, "the job's claude to be gone after the cancel")
    world.spawn_json("teardown", aid, ok=(0, 20))
