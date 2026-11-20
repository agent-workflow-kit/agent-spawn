"""agent-spawn's hooks, ``wait`` exit codes and run timestamps against the real claude-session,
wtctl, termctl and tmux, with a stub ``claude`` (wave 2 of #100; no Claude session, no billing).

The stub is the only fake: claude-session runs it like the real binary, so everything that depends
on claude-session's real outcome mapping (a report filed through ``claude-session report`` is
ingested when the stub exits, a ``--timeout`` ends a run ``timeout``, a SIGKILLed wrapper leaves
the run ``lost``) is the real tool's. What the stub does not show (real Claude's session timing) is listed in
docs/live-testing.md, "Not covered".

Human-only flags (``--override-escalation``) are driven the way the harness drives every
human-only gate: the children have no TTY on stdin, so the test passes an explicit
``--actor human:<user>``.
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import json
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from .billed import pid_gone
from .harness import World, check, process_token, wait_for
from .test_agent_spawn_stub import agent_of, show, stub_setup, wait_status
from .test_wtctl import Wt

SC = pytest.mark.live_scenario
HUMAN = ("--actor", "human:tester")


def write_script(world: World, name: str, body: str) -> Path:
    path = world.root / "hooks" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o700)
    return path


def hook_config(kind: str, *entries: tuple[Path, int | None]) -> str:
    """The ``hooks:`` YAML of one hook name: each entry a script path and an optional timeout."""
    lines = ["hooks:\n", f"  {kind}:\n"]
    for script, timeout in entries:
        lines.append(f'    - cmd: "{script}"\n')
        if timeout is not None:
            lines.append(f"      timeout: {timeout}s\n")
    return "".join(lines)


def launch_args(wt: Wt, *extra: str) -> list[str]:
    return [
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
        *extra,
    ]


def nothing_created(world: World, what: str, known: tuple[str, ...] = ()) -> None:
    """A launch stopped by agent-spawn's own ``pre-launch`` hook created no agent, no session, no
    private prompt copy and started no claude (Launch order step 2: "nothing is created and no
    agent record exists")."""
    check(
        [a["agent_id"] for a in world.spawn_json("list")["agents"]] == list(known),
        f"{what}: no agent record exists (beyond {list(known)})",
        "interface.md Launch order step 2",
    )
    check(
        len(world.json_tool("claude-session", "list")["sessions"]) == len(known),
        f"{what}: no claude-session session was allocated (beyond {len(known)})",
        "interface.md Launch order step 2",
    )
    prompts = world.state / "agent-spawn" / "prompts"
    check(
        not prompts.exists() or not any(prompts.iterdir()),
        f"{what}: the prompt copies of step 1 were deleted",
        "interface.md Launch order step 1 (any exit before step 3 deletes the directory)",
        list(prompts.iterdir()) if prompts.exists() else [],
    )
    if not known:
        check(
            not world.claude_pids.exists(),
            f"{what}: the stub claude never ran",
            "interface.md Launch order step 2",
        )


@SC("as-prelaunch-hooks")
def test_agent_spawn_pre_launch_hook_veto_escalation_timeout_and_override(world: World) -> None:
    """interface.md Hooks (``pre-launch``: veto exits 11, escalation exits 13, a timeout vetoes, an
    escalation does not stop the later hooks, a later veto overrides an earlier escalation) and
    Escalation overrides (``launch --override-escalation`` is human-only, overrides only
    agent-spawn's own ``pre-launch`` escalation, and the payload says so), Launch order step 2
    (nothing is created), the error envelope's ``hook`` and ``hook_stderr``. #100 (pre-launch hook
    veto and timeout), #36 #78 (override-escalation)."""
    payload = world.root / "payload.json"
    ran = world.root / "ran.log"
    veto = write_script(
        world,
        "veto.sh",
        f"cat > {payload}; echo ran-veto >> {ran}; echo 'blocked by policy' >&2; exit 1\n",
    )
    escalate = write_script(
        world,
        "escalate.sh",
        f"cat > {payload}; echo ran-escalate >> {ran}; echo 'needs a human' >&2; exit 13\n",
    )
    slow = write_script(
        world,
        "slow.sh",
        f"cat > /dev/null; echo $$ > {world.root / 'slow.pid'}; echo ran-slow >> {ran}; sleep 120\n",
    )
    wt = stub_setup(world, "exit 0", agent_spawn_config=hook_config("pre-launch", (veto, None)))

    # a veto: exit 11, the envelope names the hook and carries its stderr, nothing was created
    refused = world.spawn_json(*launch_args(wt), ok=(11,))
    error = refused["error"]
    world.observe("agent-spawn.pre-launch.veto", error)
    check(
        error["code"] == "vetoed"
        and error["agent_id"] is None
        and "blocked by policy" in (error["hook_stderr"] or ""),
        "a pre-launch veto exits 11 with code vetoed, no agent id and the hook's stderr",
        "interface.md Hooks (pre-launch veto exits 11), Error envelope (hook, hook_stderr)",
        error,
    )
    body = json.loads(payload.read_text())
    check(
        body["hook"] == "pre-launch"
        and body["tool"] == "agent-spawn"
        and body["object_id"] is None
        and body["override_escalation"] is False
        and body["reads_third_party"] is False,
        "the pre-launch payload names the hook, has no object id and override_escalation false",
        "interface.md Hooks (pre-launch payload: object_id null in pre-launch)",
        {
            k: body.get(k)
            for k in ("hook", "tool", "object_id", "override_escalation", "reads_third_party")
        },
    )
    nothing_created(world, "after a veto")

    # an escalation: exit 13, code escalated
    world.write_config(
        "agent-spawn",
        "auto_report_resume: false\nhandoff_timeout: 30s\n"
        + hook_config("pre-launch", (escalate, None)),
    )
    escalated = world.spawn_json(*launch_args(wt), ok=(13,))
    check(
        escalated["error"]["code"] == "escalated"
        and "needs a human" in (escalated["error"]["hook_stderr"] or "")
        and escalated["error"]["agent_id"] is None,
        "a pre-launch escalation exits 13 with code escalated and no agent",
        "interface.md Hooks (escalation exits 13)",
        escalated["error"],
    )
    nothing_created(world, "after an escalation")

    # the human-only flag without a human actor and no TTY: exit 2, hook not run again
    runs_before = ran.read_text().count("ran-escalate")
    refused_flag = world.spawn_json(*launch_args(wt, "--override-escalation"), ok=(2,))
    check(
        refused_flag["error"]["code"] == "usage"
        and ran.read_text().count("ran-escalate") == runs_before,
        "--override-escalation without a TTY or a human actor exits 2 before any hook runs",
        "interface.md Global flags (Human-only flags)",
        refused_flag["error"],
    )
    nothing_created(world, "after the flag refusal without a human actor")
    non_human = world.spawn_json(
        *launch_args(wt, "--override-escalation", "--actor", "auto:bot"), ok=(2,)
    )
    check(
        non_human["error"]["code"] == "usage",
        "an explicit non-human actor is refused for --override-escalation",
        "interface.md Global flags (an explicit non-human --actor is refused)",
        non_human["error"],
    )
    nothing_created(world, "after the flag refusal with a non-human actor")

    # with a human actor the escalation counts as allow; the payload says so; the launch runs
    done = world.spawn_json(*launch_args(wt, "--override-escalation", *HUMAN, "--wait"), ok=(14,))
    check(
        json.loads(payload.read_text())["override_escalation"] is True,
        "the pre-launch payload of an overriding invocation has override_escalation true",
        "interface.md Escalation overrides (every pre-hook payload ... override_escalation: true)",
    )
    overridden_id = agent_of(done)["agent_id"]
    check(
        agent_of(done)["status"] == "parked" and done["outcome"] == "exited-unreported",
        "with the override the launch went ahead (the stub exited 0: exited-unreported, 14)",
        "interface.md Escalation overrides (exit 13 from any of them counts as allow)",
        {k: done.get(k) for k in ("outcome", "exit_code")},
    )

    # an escalation does not stop later hooks; a later veto overrides it and still vetoes under override
    ran.write_text("")
    world.write_config(
        "agent-spawn",
        "auto_report_resume: false\nhandoff_timeout: 30s\n"
        + hook_config("pre-launch", (escalate, None), (veto, None)),
    )
    both = world.spawn_json(*launch_args(wt), ok=(11,))
    check(
        both["error"]["code"] == "vetoed"
        and ran.read_text().split() == ["ran-escalate", "ran-veto"],
        "an escalating hook does not stop the next hook, and the later veto wins (exit 11)",
        "interface.md Hooks Execution (an exit 13 escalates but the later hooks still run; a later veto overrides)",
        (both["error"]["code"], ran.read_text()),
    )
    nothing_created(world, "after a later veto over an escalation", (overridden_id,))
    still = world.spawn_json(*launch_args(wt, "--override-escalation", *HUMAN), ok=(11,))
    check(
        still["error"]["code"] == "vetoed",
        "--override-escalation does not override a veto",
        "interface.md Escalation overrides (a veto still vetoes)",
        still["error"],
    )
    nothing_created(world, "after --override-escalation did not lift a veto", (overridden_id,))

    # a timeout vetoes; the hook's process group is ended
    ran.write_text("")
    world.write_config(
        "agent-spawn",
        "auto_report_resume: false\nhandoff_timeout: 30s\n" + hook_config("pre-launch", (slow, 1)),
    )
    timed_out = world.spawn_json(*launch_args(wt), ok=(11,), timeout=90)
    world.observe("agent-spawn.pre-launch.timeout", timed_out["error"])
    check(
        timed_out["error"]["code"] == "vetoed",
        "a pre-launch hook that times out vetoes (exit 11)",
        "interface.md Hooks Execution (a timeout vetoes)",
        timed_out["error"],
    )
    pid = int((world.root / "slow.pid").read_text())
    wait_for(lambda: pid_gone(pid), 20, "the timed-out hook's process to be ended")
    nothing_created(world, "after a timeout", (overridden_id,))


CS_STUB = """\
mode=$(cat {mode})
case "$mode" in
  unreported) exit 0 ;;
  failed) echo "stub failed" | claude-session report failed --reason-file - > /dev/null; exit 0 ;;
  blocked) echo "stub blocked" | claude-session report blocked --reason-file - > /dev/null; exit 0 ;;
  needs-input) echo "stub asks" | claude-session report needs-input --reason-file - > /dev/null; exit 0 ;;
  done) claude-session report done --summary "stub done" > /dev/null; exit 0 ;;
  loop) while :; do sleep 0.2; done ;;
  *) exit 3 ;;
esac
"""


def timestamps(shown: dict[str, Any]) -> list[str | None]:
    return [r["adopted_at"] for r in shown["runs"]]


def is_date_time(text: Any) -> bool:
    try:
        datetime.fromisoformat(str(text))
    except ValueError:
        return False
    return isinstance(text, str)


@SC("as-launch-phase-hooks")
def test_claude_session_pre_launch_veto_timeout_escalation_and_adopted_at(world: World) -> None:
    """The launch phase of a run (claude-session's ``pre-launch`` hook, interface.md Outcome table
    rows ``vetoed`` and ``escalated`` and Supervision/Launch order step 8): a veto ends the run
    ``vetoed``/``hook_veto`` (11), a timeout ``hook_timeout`` (11), an escalation ``escalated``
    (13, parked ``escalated`` by ``claude-session``); ``wait`` repeats the code; ``show`` reports
    ``adopted_at`` null for a run that ended in its launch phase and a date-time for the run that
    started (#91, #100: "null for a run vetoed in its launch phase"); ``resume
    --override-escalation`` goes to claude-session only for its own escalation (Escalation
    overrides) and is human-only."""
    gate = world.root / "gate.mode"
    gate.write_text("veto")
    slow_pid = world.root / "cs-slow.pid"
    script = write_script(
        world,
        "cs-gate.sh",
        f"""cat > /dev/null
case "$(cat {gate})" in
  veto) echo 'cs says no' >&2; exit 1 ;;
  escalate) echo 'cs wants a human' >&2; exit 13 ;;
  slow) echo $$ > {slow_pid}; sleep 120 ;;
  *) exit 0 ;;
esac
""",
    )
    wt = stub_setup(world, "exit 0")
    # a generous timeout for the hooks that must not time out (a loaded host must not turn a plain
    # exit 1 into hook_timeout); only the deliberately slow case below uses 1 s
    world.claude_session_config("lost_grace: 120s\n" + hook_config("pre-launch", (script, 10)))

    vetoed = world.spawn_json(*launch_args(wt, "--wait"), ok=(11,))
    aid = agent_of(vetoed)["agent_id"]
    run = vetoed["run"]
    world.observe(
        "agent-spawn.cs-veto",
        {k: vetoed.get(k) for k in ("phase", "outcome", "end_reason", "exit_code")},
    )
    check(
        vetoed["phase"] == "launch"
        and vetoed["outcome"] == "vetoed"
        and vetoed["end_reason"] == "hook_veto"
        and run["agent_id"] == aid,
        "a claude-session pre-launch veto ends the run in the launch phase: vetoed / hook_veto, exit 11",
        "interface.md Outcome table (vetoed 11); claude-session interface.md Outcome mapping (launch phase)",
        {k: vetoed.get(k) for k in ("phase", "outcome", "end_reason", "exit_code")},
    )
    parked = show(world, aid)
    check(
        parked["agent"]["status"] == "parked"
        and parked["agent"]["park_reason"] == "vetoed"
        and parked["agent"]["needs_attention"] is None,
        "the agent parks vetoed without attention",
        "interface.md Outcome table (vetoed 11: parked, none, vetoed)",
        {k: parked["agent"][k] for k in ("status", "park_reason", "needs_attention")},
    )
    check(
        timestamps(parked) == [None],
        "adopted_at is null for a run that was vetoed in its launch phase",
        "interface.md show (null for a run ... vetoed or ended in its launch phase); #91; #100",
        timestamps(parked),
    )
    check(
        not world.claude_pids.exists(),
        "the stub claude was never spawned by a vetoed launch",
        "claude-session interface.md Launch order (the hook runs before claude is spawned)",
    )
    waited = world.spawn_json("wait", aid, ok=(11,))
    check(
        waited["outcome"] == "vetoed" and waited["exit_code"] == 11,
        "wait of the vetoed agent exits 11 with the vetoed outcome",
        "interface.md Exit codes (11 for waiting commands: the run's launch was vetoed)",
        waited.get("outcome"),
    )

    # a timeout vetoes (hook_timeout)
    gate.write_text("slow")
    world.claude_session_config("lost_grace: 120s\n" + hook_config("pre-launch", (script, 1)))
    slow = world.spawn_json(*launch_args(wt, "--wait"), ok=(11,), timeout=90)
    check(
        slow["outcome"] == "vetoed" and slow["end_reason"] == "hook_timeout",
        "a claude-session pre-launch hook that times out ends the run vetoed / hook_timeout (11)",
        "claude-session interface.md Hooks (timeout: veto with end reason hook_timeout)",
        {k: slow.get(k) for k in ("outcome", "end_reason")},
    )
    check(
        timestamps(show(world, agent_of(slow)["agent_id"])) == [None],
        "adopted_at is null after a hook_timeout too",
        "interface.md show (launch phase end)",
    )
    slow_hook = int(slow_pid.read_text())
    wait_for(lambda: pid_gone(slow_hook), 20, "the timed-out claude-session hook's process to end")
    check(
        world.spawn_json("wait", agent_of(slow)["agent_id"], ok=(11,))["end_reason"]
        == "hook_timeout",
        "wait of the hook_timeout run exits 11 again with end reason hook_timeout",
        "interface.md Exit codes (11 for waiting commands)",
    )
    world.claude_session_config("lost_grace: 120s\n" + hook_config("pre-launch", (script, 10)))

    # an escalation parks the agent escalated by claude-session (exit 13)
    gate.write_text("escalate")
    esc = world.spawn_json(*launch_args(wt, "--wait"), ok=(13,))
    eid = agent_of(esc)["agent_id"]
    shown = show(world, eid)["agent"]
    check(
        esc["outcome"] == "escalated"
        and esc["end_reason"] == "hook_escalate"
        and shown["park_reason"] == "escalated"
        and shown["escalated_by"] == "claude-session"
        and shown["needs_attention"] == "escalated",
        "a claude-session pre-launch escalation exits 13: parked escalated, escalated_by claude-session, attention escalated",
        "interface.md Outcome table (escalated 13, phase launch: escalated_by claude-session)",
        {k: shown[k] for k in ("park_reason", "escalated_by", "needs_attention")},
    )
    check(
        world.spawn_json("wait", eid, ok=(13,))["exit_code"] == 13,
        "wait of the escalated launch exits 13",
        "interface.md Exit codes (13 for waiting commands: its launch escalated)",
    )

    # resume --override-escalation: human-only, then goes to claude-session and the run starts
    refused = world.spawn_json(
        "resume", eid, "--override-escalation", "--mode", "headless", "--prompt", "ok", ok=(2,)
    )
    check(
        refused["error"]["code"] == "usage",
        "resume --override-escalation without a human actor exits 2",
        "interface.md Global flags (Human-only flags)",
        refused["error"],
    )
    # a vetoed agent has no escalation to override (escalated_by null): the flag exits 2 for it
    no_target = world.spawn_json(
        "resume",
        aid,
        "--override-escalation",
        *HUMAN,
        "--mode",
        "headless",
        "--prompt",
        "ok",
        ok=(2,),
    )
    check(
        no_target["error"]["code"] == "usage",
        "resume --override-escalation of an agent whose last run did not end escalated exits 2 (no target)",
        "interface.md Escalation overrides (without a target the flag exits 2)",
        no_target["error"],
    )
    resumed = world.spawn_json(
        "resume",
        eid,
        "--override-escalation",
        *HUMAN,
        "--mode",
        "headless",
        "--prompt",
        "ok",
        "--wait",
        ok=(14,),
    )
    check(
        resumed["run"]["agent_id"] == eid
        and resumed["agent"]["run"] == 2
        and resumed["outcome"] == "exited-unreported",
        "with the override the escalated launch continues: run 2 starts (the stub exits 0: 14)",
        "interface.md Escalation overrides (claude-session start --resume --override-escalation, only when the last run ended escalated)",
        {k: resumed.get(k) for k in ("outcome", "exit_code")},
    )
    after = show(world, eid)
    first, second = timestamps(after)
    check(
        first is None and is_date_time(second),
        "run 1 (escalated in its launch phase) has adopted_at null; run 2 (started) a date-time",
        "interface.md show (adopted_at: the run's Started time); #91",
        timestamps(after),
    )
    world.observe("agent-spawn.adopted_at.resumed-after-escalation", timestamps(after))
    # the agent has a session, but its last run ended exited-unreported, not escalated: no target
    not_escalated = world.spawn_json(
        "resume",
        eid,
        "--override-escalation",
        *HUMAN,
        "--mode",
        "headless",
        "--prompt",
        "ok",
        ok=(2,),
    )
    check(
        not_escalated["error"]["code"] == "usage",
        "resume --override-escalation of an agent with a session whose last run did not end escalated exits 2",
        "interface.md Escalation overrides (claude-session's override only when the last run ended escalated)",
        not_escalated["error"],
    )
    log = world.state / "claude-session" / "audit.jsonl"
    audits = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    world.observe(
        "claude-session.audit.override",
        [a for a in audits if "override" in json.dumps(a)][:2],
    )


@SC("as-wait-exit-codes")
def test_wait_and_launch_wait_exit_the_outcome_code_of_every_outcome(world: World) -> None:
    """interface.md Exit codes (14 to 18 for waiting commands) and Outcome table, against
    claude-session's real outcome mapping: ``exited-unreported`` 14 (stub exits 0), ``killed`` 15
    (``stop``), ``timeout`` 16 (``--timeout``, a ``pending_cause``), ``lost`` 17 (exec SIGKILLed
    while the child dies with it), ``failed`` 18 and ``blocked``/``needs-input`` 13 (a real
    ``claude-session report`` the stub files, ingested when it exits), ``done`` 0; ``wait
    --timeout`` 22 while live; ``wait`` repeats each code and the result object; unknown id 10.
    #100 (``wait`` exit codes 14/16/17/18), #36 #8 claude-session (exit codes by outcome)."""
    mode = world.root / "mode"
    wt = stub_setup(world, CS_STUB.format(mode=mode))
    cases: list[tuple[str, int, str, str | None]] = [
        # mode, exit, outcome, park_reason
        ("unreported", 14, "exited-unreported", "unreported"),
        ("failed", 18, "failed", "failed"),
        ("blocked", 13, "blocked", "blocked"),
        ("needs-input", 13, "needs-input", "escalated"),
        ("done", 0, "done", "done"),
    ]
    for name, code, outcome, park in cases:
        mode.write_text(name)
        done = world.spawn_json(*launch_args(wt, "--wait"), ok=(code,))
        aid = agent_of(done)["agent_id"]
        check(
            done["outcome"] == outcome and done["exit_code"] == code,
            f"a stub run ending {name} makes launch --wait exit {code} with outcome {outcome}",
            "interface.md Outcome table / Exit codes",
            {k: done.get(k) for k in ("outcome", "end_reason", "exit_code")},
        )
        waited = world.spawn_json("wait", aid, ok=(code,))
        check(
            waited["outcome"] == outcome and agent_of(waited)["park_reason"] == park,
            f"wait of the parked {name} agent exits {code} again and the agent parked {park}",
            "interface.md wait (a parked agent returns at once with its last outcome)",
            (waited.get("outcome"), agent_of(waited)["park_reason"]),
        )
        world.observe(
            f"agent-spawn.wait.{name}",
            {k: done.get(k) for k in ("outcome", "end_reason", "exit_code")},
        )

    # timeout: claude-session's --timeout is a pending_cause; the stub dies of SIGTERM
    mode.write_text("loop")
    timed = world.spawn_json(*launch_args(wt, "--wait", "--timeout", "3s"), ok=(16,), timeout=120)
    taid = agent_of(timed)["agent_id"]
    check(
        timed["outcome"] == "timeout" and timed["end_reason"] == "timeout",
        "claude-session's --timeout ends a run timeout / timeout (16)",
        "claude-session interface.md Outcome mapping (pending_cause timeout); interface.md Exit codes (16)",
        {k: timed.get(k) for k in ("outcome", "end_reason")},
    )
    check(
        agent_of(world.spawn_json("wait", taid, ok=(16,)))["park_reason"] == "timeout",
        "wait repeats 16 and the agent parked timeout",
        "interface.md Outcome table (timeout 16)",
    )

    # a live agent: wait --timeout expires with 22 and the agent keeps running
    live = world.spawn_json(*launch_args(wt))
    laid = agent_of(live)["agent_id"]
    world.note_session(agent_of(live)["csession_id"])
    state = wait_status(world, laid, "live")
    world.track(int(state["supervisor"]["pid"]), tree=True)
    stuck = world.spawn_json("wait", laid, "--timeout", "2s", ok=(22,))
    check(
        stuck["timed_out"] is True
        and stuck["run"] is None
        and show(world, laid)["agent"]["status"] == "live",
        "wait --timeout on a live agent exits 22 (timed_out true, run null) and leaves the agent live",
        "interface.md wait (--timeout expiring first exits 22)",
        {k: stuck.get(k) for k in ("timed_out", "run")},
    )
    # killed: stop parks 15
    world.spawn_json("stop", laid, "--grace", "2s")
    killed = world.spawn_json("wait", laid, ok=(15,))
    check(
        killed["outcome"] == "killed" and agent_of(killed)["park_reason"] == "killed",
        "wait of a stopped run exits 15 (killed)",
        "interface.md Outcome table (killed 15, stopped)",
        killed.get("outcome"),
    )

    # lost: SIGKILL of exec while the child dies with its wrapper (PDEATHSIG): run lost, 17
    lost_live = world.spawn_json(*launch_args(wt))
    xid = agent_of(lost_live)["agent_id"]
    world.note_session(agent_of(lost_live)["csession_id"])
    state = wait_status(world, xid, "live")
    world.track(int(state["supervisor"]["pid"]), tree=True)
    exec_pid = int(wait_for(lambda: show(world, xid)["supervisor"]["exec_pid"], 20, "the exec pid"))
    token = process_token(exec_pid)
    check(token is not None, "exec is a live process", "interface.md Supervision")
    # the stub child of this run: tracked by pid plus start-time token, so teardown ends it even if a check fails
    child_pid = int(
        wait_for(
            lambda: (
                world.json_tool("claude-session", "status", agent_of(lost_live)["csession_id"])
                .get("run", {})
                .get("child_pid")
            ),
            30,
            "the stub child's pid",
        )
    )
    child_token = process_token(child_pid)
    if child_token is not None:
        world.pids.append((child_pid, child_token))
    if process_token(exec_pid) == token:  # identity re-read immediately before the signal
        os.kill(exec_pid, signal.SIGKILL)  # the exact exec pid agent-spawn recorded
    lost = world.spawn_json("wait", xid, "--timeout", "90s", ok=(17,), timeout=120)
    check(
        lost["outcome"] == "lost"
        and agent_of(lost)["park_reason"] == "lost"
        and agent_of(lost)["needs_attention"] == "lost",
        "a headless run whose wrapper was SIGKILLed with a SIGTERM-honoring child ends lost: wait exits 17, parked lost with attention lost",
        "interface.md Outcome table (lost 17); #36 #75 3 (First billed run: exec-only SIGKILL)",
        {k: lost.get(k) for k in ("outcome", "end_reason")},
    )

    # an unknown id
    unknown = world.agent_spawn("wait", "ag-zzzzzzzz", "--json")
    check(
        unknown.returncode == 10,
        "wait of an unknown id exits 10",
        "interface.md wait (exit 10 only for an unknown id or a tombstone without a run)",
        unknown.returncode,
    )


RESULT_LINE = '{"type":"result","subtype":"success","is_error":false,"result":"ok","num_turns":1,"total_cost_usd":0}'

ADOPTED_STUB = """\
mode=$(cat {root}/amode)
case "$mode" in
  loop) while :; do sleep 0.2; done ;;
  done) echo "stub done" | claude-session report done --summary-file - > /dev/null; exit 0 ;;
  autoresume)
    if [ -e {root}/auto-mark ]; then
      echo "stub done" | claude-session report done --summary-file - > /dev/null
    else
      touch {root}/auto-mark
      echo '{result}'
    fi
    exit 0 ;;
  promote)
    if [ -e {root}/promote-mark ]; then
      while :; do sleep 0.2; done
    fi
    touch {root}/promote-mark
    echo "stub asks" | claude-session report needs-input --reason-file - > /dev/null
    exit 0 ;;
esac
"""


def parsed(runs: list[dict[str, Any]], key: str = "adopted_at") -> list[datetime]:
    return [datetime.fromisoformat(r[key]) for r in runs]


@SC("as-adopted-at")
def test_adopted_at_is_set_for_launched_resumed_auto_resumed_and_promoted_runs(
    world: World,
) -> None:
    """#91 / #100: ``show --json`` ``runs[].adopted_at`` is a date-time for a launched run, a
    resumed run, the automatic report resume and a promoted run (the run's Started time, written
    once by the supervisor when claude-session reports ``phase: session``), never earlier than the
    agent's creation and not later than the run's end, and never changed by a later run. The stub
    stands in for ``claude`` (its stream-json ``result`` line makes ``--no-implicit-done`` end
    ``no_report``, which is what the automatic report resume answers); the null case of a
    launch-phase veto is in ``as-launch-phase-hooks``. What this does NOT show is the timing of a
    real Claude start (the billed candidates in docs/live-testing.md, Not covered)."""
    root = world.root
    mode = root / "amode"
    wt = stub_setup(
        world,
        ADOPTED_STUB.format(root=root, result=RESULT_LINE),
    )
    # the automatic report resume on, and implement runs not promoted (rewritten once, not twice)
    world.write_config(
        "agent-spawn",
        "auto_report_resume: true\nhandoff_timeout: 30s\nkinds:\n  implement:\n    promote: false\n",
    )

    # launched, then resumed
    mode.write_text("loop")
    live = world.spawn_json(*launch_args(wt))
    aid = agent_of(live)["agent_id"]
    world.note_session(agent_of(live)["csession_id"])
    state = wait_status(world, aid, "live")
    world.track(int(state["supervisor"]["pid"]), tree=True)
    runs = state["runs"]
    check(
        len(runs) == 1 and is_date_time(runs[0]["adopted_at"]) and runs[0]["ended_at"] is None,
        "a live launched run already has a date-time adopted_at and no end",
        "interface.md show (adopted_at: the run's Started time); #91",
        runs,
    )
    first = runs[0]["adopted_at"]
    created = state["agent"]["created"]
    check(
        datetime.fromisoformat(created) <= datetime.fromisoformat(first),
        "adopted_at is not earlier than the agent's creation",
        "interface.md show; #91",
        (created, first),
    )
    world.spawn_json("stop", aid, "--grace", "2s")
    wait_status(world, aid, "parked")
    mode.write_text("done")
    resumed = world.spawn_json("resume", aid, "--mode", "headless", "--prompt", "ok", "--wait")
    after = show(world, aid)
    stamps = parsed(after["runs"])
    check(
        resumed["agent"]["run"] == 2
        and len(stamps) == 2
        and after["runs"][0]["adopted_at"] == first
        and stamps[0] <= stamps[1],
        "a resumed run has its own date-time adopted_at; run 1's value is unchanged (written once)",
        "interface.md show (written once and never changed, for every run of the chain including a resume)",
        timestamps(after),
    )
    check(
        stamps[0] <= datetime.fromisoformat(after["runs"][0]["ended_at"]) <= stamps[1],
        "run 1 ended between its adoption and the next run's",
        "interface.md show",
        [(r["adopted_at"], r["ended_at"]) for r in after["runs"]],
    )

    # the automatic report resume: headless, --no-implicit-done, a result but no report
    mode.write_text("autoresume")
    auto = world.spawn_json(*launch_args(wt, "--wait", "--no-implicit-done"), ok=(0,), timeout=120)
    auto_show = show(world, agent_of(auto)["agent_id"])
    flags = [(r["run"], r["auto_resumed"], r["outcome"]) for r in auto_show["runs"]]
    world.observe("agent-spawn.adopted_at.auto-resume", flags)
    check(
        len(auto_show["runs"]) == 2
        and auto_show["runs"][1]["auto_resumed"] is True
        and all(is_date_time(r["adopted_at"]) for r in auto_show["runs"])
        and parsed(auto_show["runs"])[0] <= parsed(auto_show["runs"])[1],
        "the first run ended no_report, the automatic report resume is run 2 and both runs have a date-time adopted_at, in order",
        "interface.md show (... the automatic report resume); Outcome table (exited-unreported headless: automatic report resume once); #91",
        (flags, timestamps(auto_show)),
    )

    # a promoted run: headless needs-input, promotion into a window
    mode.write_text("promote")
    promoted = world.spawn_json(*launch_args(wt, "--promote"))
    pid_ = agent_of(promoted)["agent_id"]
    world.note_session(agent_of(promoted)["csession_id"])

    def promoted_live() -> dict[str, Any] | None:
        shown = show(world, pid_)
        runs_ = shown["runs"]
        if len(runs_) >= 2 and runs_[1]["adopted_at"] and shown["agent"]["status"] == "live":
            return shown
        return None

    shown = wait_for(promoted_live, 90, "the promoted run to be live in its window")
    sup = shown["supervisor"]
    world.track(int(sup["pid"]), tree=True)
    check(
        shown["runs"][1]["promoted"] is True
        and shown["runs"][1]["mode"] == "interactive"
        and is_date_time(shown["runs"][0]["adopted_at"])
        and parsed(shown["runs"])[0] <= parsed(shown["runs"])[1],
        "the promoted run (run 2, interactive) has a date-time adopted_at after run 1's",
        "interface.md show (... and a promoted run); Promotion; #91",
        timestamps(shown),
    )
    world.observe("agent-spawn.adopted_at.promoted", timestamps(shown))
    world.spawn_json("stop", pid_, "--grace", "2s")
    wait_status(world, pid_, "parked", 60)
