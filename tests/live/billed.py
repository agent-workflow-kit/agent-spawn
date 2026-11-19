"""Helpers of the billed scenarios: every launch is ``claude --model haiku`` with a tiny prompt.

Cost bounds (see docs/live-testing.md): the ``claude`` wrapper counts launches against
``MAX_LIVE_SESSIONS`` and refuses a non-haiku model; headless launches carry ``--max-turns`` and
``--timeout``; every test has a wall-clock bound; interactive sessions end by the test closing the
window or by claude-session ``stop`` at the latest (teardown kills the private tmux server).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .harness import LiveError, World, process_token, wait_for
from .test_wtctl import Wt

PROMPT_OK = "Reply with the single word ok."
#: A tiny headless run takes at least three turns (read the prompt file, run the report command,
#: the final answer) and claude gets ``--max-turns`` + 1 (the Stop backstop turn), so 6 leaves room.
MAX_TURNS = "6"
HEADLESS_BOUNDS = ["--model", "haiku", "--max-turns", MAX_TURNS, "--timeout", "4m"]
#: ``--no-auto-exit``: the session stays up after its turn, so a test acts on a settled idle
#: session instead of racing claude-session's auto-exit.
INTERACTIVE_BOUNDS = ["--model", "haiku", "--timeout", "10m", "--no-auto-exit"]


def bsetup(world: World, *, cs_extra: str = "", spawn_extra: str = "") -> Wt:
    """Private claude-session and agent-spawn config (the counting claude wrapper, auth names), a
    wtctl repo (also the trusted cwd of non-worktree runs) and the private ``~/.claude.json``."""
    world.claude_session_config(cs_extra)
    world.write_config("agent-spawn", "handoff_timeout: 30s\n" + spawn_extra)
    wt = Wt(world)
    world.trust(wt.repo.clone)
    return wt


def spawn(world: World, *args: str, ok: tuple[int, ...] = (0,), timeout: float = 240) -> Any:
    """``agent-spawn <args> --json`` with the credentials in its environment (the handoff and
    claude-session pass them on to claude)."""
    return world.spawn_json(*args, ok=ok, auth=True, timeout=timeout)


def spawn_raw(world: World, *args: str, timeout: float = 240) -> Any:
    return world.agent_spawn(*args, "--json", auth=True, timeout=timeout)


def launch_headless(
    world: World, wt: Wt, prompt: str = PROMPT_OK, *extra: str, ok: tuple[int, ...] = (0,)
) -> dict[str, Any]:
    done = spawn(
        world,
        "launch",
        "--mode",
        "headless",
        "--prompt",
        prompt,
        *HEADLESS_BOUNDS,
        "--wait",
        *extra,
        ok=ok,
    )
    adopt(world, done["agent"])
    return done


def adopt(world: World, agent: dict[str, Any]) -> None:
    """Make teardown end this agent's session (``claude-session stop``) and kill what it left."""
    world.note_session(agent.get("csession_id") or "")


def track_agent(world: World, agent_id: str) -> dict[str, Any]:
    """Track the live agent's supervisor and its current descendants (supervisor, exec, claude)
    for teardown, as soon as it is live: a launch without ``--wait`` or a test that fails early
    must not leave a detached supervisor or claude running."""
    state = wait_agent(
        world,
        agent_id,
        lambda s: s["agent"]["status"] == "live" and bool((s.get("supervisor") or {}).get("pid")),
        f"agent {agent_id} to be live with a supervisor",
        120,
    )
    world.track(int(state["supervisor"]["pid"]), tree=True)
    adopt(world, state["agent"])
    return state


def launch_window(world: World, wt: Wt, prompt: str = PROMPT_OK, *extra: str) -> dict[str, Any]:
    launched = launch_window_raw(world, wt, prompt, *extra)
    adopt(world, launched["agent"])
    return launched


def launch_window_raw(world: World, wt: Wt, prompt: str = PROMPT_OK, *extra: str) -> dict[str, Any]:
    return spawn(
        world,
        "launch",
        "--mode",
        "interactive",
        "--prompt",
        prompt,
        *INTERACTIVE_BOUNDS,
        "--worktree",
        "none",
        "--cwd",
        str(wt.repo.clone),
        *extra,
    )


def show(world: World, agent_id: str) -> dict[str, Any]:
    return world.spawn_json("show", agent_id, auth=False)


def wait_agent(
    world: World,
    agent_id: str,
    predicate: Callable[[dict[str, Any]], bool],
    what: str,
    timeout: float = 120,
) -> dict[str, Any]:
    box: dict[str, Any] = {}

    def done() -> bool:
        box["show"] = show(world, agent_id)
        return predicate(box["show"])

    wait_for(done, timeout, what, interval=1.0)
    return box["show"]


def wait_session_state(
    world: World, agent_id: str, states: set[str], timeout: float = 150
) -> dict[str, Any]:
    return wait_agent(
        world,
        agent_id,
        lambda s: (s.get("session") or {}).get("state") in states,
        f"agent {agent_id} session state in {sorted(states)}",
        timeout,
    )


def wait_turn_done(world: World, agent_id: str, timeout: float = 180) -> dict[str, Any]:
    """Wait until the interactive run has had its turn: ``first_working_at`` is set and the
    session state is settled (idle, reported or needs-input); then require idle or reported."""
    state = wait_agent(
        world,
        agent_id,
        lambda s: (
            bool(s["runs"] and s["runs"][0].get("first_working_at"))
            and (s.get("session") or {}).get("state") in {"idle", "reported", "needs-input"}
        ),
        f"agent {agent_id} to finish its first turn",
        timeout,
    )
    got = state["session"]["state"]
    if got not in {"idle", "reported"}:
        raise LiveError(
            f"assumption broken: the one-turn prompt settles idle or reported, not {got} "
            f"[design.md D1 / #36 #27] {state['session']}"
        )
    return state


def cs_status(world: World, csession_id: str) -> dict[str, Any]:
    return world.json_tool("claude-session", "status", csession_id)


def signal_exact(world: World, pid: int, sig: int, *, comm_contains: str) -> None:
    """Signal one exact process after checking its identity (start token recorded now, command
    line names ``comm_contains``): never by pattern."""
    token = process_token(pid)
    if token is None:
        raise LiveError(f"pid {pid} is gone")
    cmdline = (
        Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    )
    if comm_contains not in cmdline:
        raise LiveError(f"pid {pid} is not {comm_contains}: {cmdline[:120]}")
    world.pids.append((pid, token))
    if process_token(pid) == token:
        os.kill(pid, sig)


def pid_gone(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat.rsplit(")", 1)[-1].split()[0] == "Z"


def expect_parked(world: World, agent_id: str, timeout: float = 90) -> dict[str, Any]:
    return wait_agent(
        world, agent_id, lambda s: s["agent"]["status"] == "parked", f"{agent_id} to park", timeout
    )["agent"]
