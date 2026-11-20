"""Helpers of the billed scenarios: every launch is ``claude --model haiku`` with a tiny prompt.

Cost bounds (see docs/live-testing.md): the ``claude`` wrapper counts launches against
``MAX_LIVE_SESSIONS`` and refuses a non-haiku model; headless launches carry ``--max-turns`` and
``--timeout``; every test has a wall-clock bound; interactive sessions end by the test closing the
window or by claude-session ``stop`` at the latest (teardown kills the private tmux server).
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .harness import LiveError, World, check, process_token, wait_for
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
    # An interactive claude also asks "Adding a new directory" for claude-session's session
    # directory (``--add-dir``, under the private state home); trust is inherited from a trusted
    # parent, so trust the directory that holds the session directories.
    sessions = prepare_state_dirs(world)
    world.trust(wt.repo.clone, sessions)
    return wt


def prepare_state_dirs(world: World) -> Path:
    """claude-session's state directories, mode 0700 like the tool makes them (chmod after mkdir
    because of the umask); returns the sessions directory."""
    base = world.home / ".local" / "state" / "claude-session"
    sessions = base / "sessions"
    for directory in (base.parent, base, sessions):
        directory.mkdir(parents=True, exist_ok=True)
    base.chmod(0o700)
    sessions.chmod(0o700)
    return sessions


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


def redact(world: World, text: str) -> str:
    """Remove credential fragments from diagnostic text: the key tail (the custom-API-key dialog
    shows it), anything shaped like an Anthropic key and the OAuth token value."""
    secrets: list[str] = []
    auth = world.auth
    if auth is not None:
        if auth.api_key:
            secrets.append(auth.api_key.strip()[-20:])
        token = auth.env.get("CLAUDE_CODE_OAUTH_TOKEN")
        if token:
            secrets.append(token)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return re.sub(r"sk-ant-[A-Za-z0-9_-]+", "[redacted]", text)


def _diagnose(world: World, agent_id: str) -> str:
    """What a timed-out interactive turn looked like: the agent's state and the pane text (the
    visible screen of the real claude TUI: no credentials)."""
    try:
        shown = show(world, agent_id)
        parts = [
            f"status={shown['agent'].get('status')} session={shown.get('session')} "
            f"first_working_at={[r.get('first_working_at') for r in shown.get('runs', [])]}"
        ]
        for pane in world.json_tool("termctl", "list", "--label", f"agent={agent_id}")["panes"]:
            cap = world.tmux(
                "capture-pane", "-p", "-t", pane["pane_id"], socket=pane["socket_path"]
            )
            parts.append(f"pane {pane['pane_id']} screen: {cap.stdout[-1500:]!r}")
        return redact(world, " | ".join(parts))
    except Exception as exc:  # diagnostics must never mask the real failure
        return f"(diagnostics failed: {exc})"


def expected_park_after_hangup(world: World, csession_id: str) -> str:
    """The park reason a closed window gives: ``done`` when the real session had already filed its
    report in its one turn ("a reported run keeps its outcome", interface.md Supervision; the
    model reports in a tiny interactive run despite ``Reply with the single word ok``), else
    ``killed``."""
    report = cs_status(world, csession_id)["run"].get("report")
    return "done" if report else "killed"


def wait_turn_done(world: World, agent_id: str, timeout: float = 180) -> dict[str, Any]:
    """Wait until the interactive run has had its turn: ``first_working_at`` is set and the
    session state is settled (idle, reported or needs-input); then require idle or reported."""
    try:
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
    except LiveError as err:
        raise LiveError(f"{err}; {_diagnose(world, agent_id)}") from err
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


def now_utc() -> datetime:
    return datetime.now(UTC)


def _stamp(value: object) -> datetime | None:
    """A ``date-time`` string parsed to an aware datetime, or None when it is not one."""
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


def check_adopted_at(
    state: dict[str, Any],
    window: tuple[datetime, datetime],
    *,
    runs: int,
    where: str,
    first: list[str] | None = None,
) -> list[str]:
    """``show --json`` ``runs[].adopted_at`` against real claude-session timing (#91, #100).

    Every one of the ``runs`` adopted runs has a date-time, non-null ``adopted_at`` that is not
    earlier than the agent's creation, is non-decreasing along the chain, never later than the
    run's own ``ended_at`` and not earlier than the previous run's ``ended_at``, and lies inside
    ``window`` (the test's wall-clock bounds, 1 s of slack for the millisecond truncation and the
    clock read order). ``first`` is the list returned by an earlier call for the same runs: those
    values must be unchanged (written once). Returns the ``adopted_at`` strings."""
    shown = state["runs"]
    slack = timedelta(seconds=1)
    check(
        len(shown) == runs,
        f"{where}: the agent has {runs} run(s)",
        "interface.md show (runs)",
        [r.get("run") for r in shown],
    )
    raw = [r.get("adopted_at") for r in shown]
    stamps = [_stamp(v) for v in raw]
    check(
        all(s is not None for s in stamps),
        f"{where}: every adopted run has a date-time adopted_at (not null)",
        "interface.md show (adopted_at: the run's Started time) / #91",
        raw,
    )
    moments = [s for s in stamps if s is not None]
    created = _stamp(state["agent"].get("created"))
    check(
        created is not None and all(created <= m for m in moments),
        f"{where}: adopted_at is not earlier than the agent's creation",
        "interface.md show; #91",
        (state["agent"].get("created"), raw),
    )
    check(
        all(a <= b for a, b in zip(moments, moments[1:], strict=False)),
        f"{where}: adopted_at is non-decreasing along the run chain",
        "interface.md show; #91",
        raw,
    )
    previous_end: datetime | None = None
    for run, moment in zip(shown, moments, strict=True):
        end = _stamp(run.get("ended_at"))
        check(
            end is None or moment <= end,
            f"{where}: run {run.get('run')} was adopted before it ended",
            "interface.md show (adopted_at is the Started moment); #91",
            (run.get("adopted_at"), run.get("ended_at")),
        )
        check(
            previous_end is None or previous_end <= moment,
            f"{where}: run {run.get('run')} was adopted after the previous run ended",
            "interface.md show; #91",
            (run.get("adopted_at"), previous_end),
        )
        previous_end = end
    check(
        all(window[0] - slack <= m <= window[1] + slack for m in moments),
        f"{where}: adopted_at lies inside the test's wall-clock window (real timing, not a stale "
        "or default value)",
        "interface.md show (wall-clock time of the supervisor's first sight); #91",
        (raw, window),
    )
    if first is not None:
        check(
            raw[: len(first)] == first,
            f"{where}: adopted_at is written once (earlier runs' values unchanged)",
            "interface.md show (written once and never changed); #91",
            (first, raw),
        )
    return [str(v) for v in raw]
