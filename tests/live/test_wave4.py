"""The unbilled live follow-ups of wave 4 (#112 to #116): hooks, termctl, wtctl and claude-session
behaviors the earlier waves left out, against the real tools with a stub ``claude`` (no Claude
session, no credentials, no billing).

Human-only flags are driven the way the harness drives every human-only gate: the children have no
TTY on stdin, so the tests pass an explicit ``--actor human:<user>`` (the one TTY case that a pty
can provide is its own scenario, ``as-human-yes-tty``).
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from .billed import pid_gone
from .harness import LiveError, World, check, process_token, wait_for
from .test_agent_spawn_hooks import HUMAN, hook_config, launch_args, write_script
from .test_agent_spawn_stub import agent_of, show, stub_setup, wait_status

SC = pytest.mark.live_scenario


def parked_stub_agent(world: World, wt: Any) -> str:
    """A parked agent from a headless stub run (``exit 0``: exited-unreported, 14)."""
    done = world.spawn_json(*launch_args(wt, "--wait"), ok=(14,))
    aid = agent_of(done)["agent_id"]
    check(
        agent_of(done)["status"] == "parked",
        "the stub run ended and the agent is parked",
        "interface.md Outcome table (exited-unreported)",
        agent_of(done),
    )
    return str(aid)


@SC("as-slow-pre-teardown")
def test_a_slow_pre_teardown_hook_times_out_and_vetoes_the_teardown(world: World) -> None:
    """interface.md Hooks (a hook that outlasts its timeout vetoes, exit 11, and its process group
    is ended) and Teardown order step 3 (a veto returns the agent to ``parked`` unchanged, and
    nothing after the hook ran); ``--override-escalation`` overrides an escalation only, never a
    timeout veto (Escalation overrides). #113."""
    ran = world.root / "ran.log"
    sleeper = world.root / "sleep.pid"
    slow = write_script(
        world,
        "slow-teardown.sh",
        f"cat > /dev/null\necho $$ > {world.root / 'hook.pid'}\necho slow >> {ran}\n"
        f"sleep 120 &\necho $! > {sleeper}\nwait\n",
    )
    wt = stub_setup(
        world, "exit 0", agent_spawn_config=hook_config("pre-teardown", (slow, 1))
    )
    aid = parked_stub_agent(world, wt)
    before = show(world, aid)["agent"]

    refused = world.spawn_json("teardown", aid, ok=(11,), timeout=90)
    world.observe("agent-spawn.pre-teardown.timeout", refused["error"])
    error = refused["error"]
    check(
        error["code"] == "vetoed" and error["hook"] == "pre-teardown",
        "a pre-teardown hook that times out vetoes the teardown: exit 11, code vetoed, hook named",
        "interface.md Hooks Execution (a timeout vetoes), Exit codes (11 a pre-teardown hook vetoed)",
        error,
    )
    for name in ("hook.pid", "sleep.pid"):
        pid = int((world.root / name).read_text())
        wait_for(lambda pid=pid: pid_gone(pid), 20, f"the timed-out hook's {name} process to end")
    after = show(world, aid)["agent"]
    check(
        after["status"] == "parked" and after["teardown_step"] is None,
        "the vetoed teardown returned the agent to parked with no teardown marker",
        "interface.md Teardown order step 3 (either returns the agent to parked unchanged)",
        {k: after[k] for k in ("status", "teardown_step")},
    )
    check(
        after["worktree"] == before["worktree"]
        and after["csession_id"] == before["csession_id"]
        and after["needs_attention"] == before["needs_attention"],
        "nothing after the hook ran: the worktree, the session and the attention are unchanged",
        "interface.md Teardown order (the hook is step 3, before any removal)",
        {"before": before, "after": after},
    )
    check(
        len(world.json_tool("claude-session", "list")["sessions"]) == 1,
        "the session was not forgotten by the vetoed teardown",
        "interface.md Teardown order step 6 (forget happens after the hooks)",
    )

    # a human override lifts an escalation only: a timeout still vetoes, and the hook ran again
    again = world.spawn_json("teardown", aid, "--override-escalation", *HUMAN, ok=(11,), timeout=90)
    check(
        again["error"]["code"] == "vetoed" and ran.read_text().count("slow") == 2,
        "--override-escalation does not lift a timeout veto: the hook ran again and vetoed again",
        "interface.md Escalation overrides (a veto still vetoes)",
        (again["error"]["code"], ran.read_text()),
    )
    for name in ("hook.pid", "sleep.pid"):
        wait_for(
            lambda name=name: pid_gone(int((world.root / name).read_text())),
            20,
            f"the second run's {name} process to end",
        )

    # a hook that finishes inside its timeout lets the teardown go on
    quick = write_script(world, "quick.sh", f"cat > /dev/null\necho quick >> {ran}\nexit 0\n")
    world.write_config(
        "agent-spawn",
        "auto_report_resume: false\nhandoff_timeout: 30s\n"
        + hook_config("pre-teardown", (quick, 5)),
    )
    torn = world.spawn_json("teardown", aid, timeout=90)
    check(
        "quick" in ran.read_text() and torn["session_forgotten"] is True,
        "a pre-teardown hook that exits 0 in time lets the teardown finish",
        "interface.md Teardown order step 3 (hooks_done marker), step 6",
        torn,
    )
    gone = world.spawn_json("show", aid)
    check(
        gone["agent"]["status"] == "torn_down",
        "the agent is torn_down after the retry",
        "interface.md Agent statuses (parked -> tearing_down -> torn_down)",
        gone["agent"]["status"],
    )


def window_launch_args(wt: Any, *extra: str) -> list[str]:
    return [
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
        *extra,
    ]


def panes_of(world: World, aid: str) -> list[dict[str, Any]]:
    return list(world.json_tool("termctl", "list", "--label", f"agent={aid}")["panes"])


@SC("as-termctl-override-escalation")
def test_termctl_open_escalation_is_overridden_by_a_human_resume(world: World) -> None:
    """interface.md Launch order step 5 (termctl exit 13 parks the agent ``escalated`` with
    ``escalated_by: termctl``, exit 11 parks it ``vetoed``) and Escalation overrides (``resume
    --override-escalation`` passes the flag to ``termctl open`` by ``escalated_by``; human-only; a
    veto still vetoes; the pre-open payload says ``override_escalation: true``; stateless: the hook
    may have changed between the two attempts). termctl interface.md Hooks, Escalation override
    and Audit (an ``--override-escalation`` run is audited). #112, #36 #78."""
    seen = world.root / "pre-open-payloads.log"
    escalate = write_script(
        world, "pre-open-escalate.sh", f"cat >> {seen}; echo >> {seen}; echo 'needs a human' >&2; exit 13\n"
    )
    wt = stub_setup(world, "while :; do sleep 0.2; done")
    world.write_config("termctl", hook_config("pre-open", (escalate, None)))

    # 1. the hook escalates: the launch parks escalated by termctl, no window exists
    escalated = world.spawn_json(*window_launch_args(wt), ok=(13,), timeout=120)
    eid = escalated["error"]["agent_id"]
    shown = show(world, eid)["agent"]
    world.observe("agent-spawn.termctl-escalation.error", escalated["error"])
    check(
        escalated["error"]["code"] == "escalated"
        and shown["status"] == "parked"
        and shown["park_reason"] == "escalated"
        and shown["escalated_by"] == "termctl"
        and shown["needs_attention"] == "escalated",
        "a termctl pre-open escalation exits 13 and parks the agent escalated with escalated_by termctl",
        "interface.md Launch order step 5 (termctl 13 -> escalated, escalated_by termctl)",
        {k: shown[k] for k in ("status", "park_reason", "escalated_by", "needs_attention")},
    )
    check(
        panes_of(world, eid) == [] and shown["window"] is None,
        "the escalated launch left no pane and the agent records no window",
        "termctl interface.md Open order step 4 (an escalation creates nothing)",
        panes_of(world, eid),
    )
    first = json.loads(seen.read_text().splitlines()[0])
    check(
        first["hook"] == "pre-open" and first["override_escalation"] is False,
        "the first pre-open payload has override_escalation false",
        "termctl interface.md Hooks (payload override_escalation)",
        {k: first.get(k) for k in ("hook", "override_escalation")},
    )

    # 2. the flag is human-only: no human actor (no TTY) and an auto actor exit 2, termctl is not asked
    runs = seen.read_text().count("pre-open")
    for actor in ((), ("--actor", "auto:bot")):
        refused = world.spawn_json("resume", eid, "--override-escalation", *actor, ok=(2,))
        check(
            refused["error"]["code"] == "usage" and seen.read_text().count("pre-open") == runs,
            f"resume --override-escalation with actor {actor or 'none'} exits 2 before termctl runs the hook again",
            "interface.md Global flags (Human-only flags); Escalation overrides",
            refused["error"],
        )
    check(
        show(world, eid)["agent"]["status"] == "parked" and panes_of(world, eid) == [],
        "the refused resumes changed nothing",
        "interface.md Escalation overrides",
    )

    # 3. a changed hook that still escalates, and the human override: the window opens
    marker = world.root / "pre-open-ran"
    changed = write_script(
        world,
        "pre-open-changed.sh",
        f"cat >> {seen}; echo >> {seen}; touch {marker}; echo 'still needs a human' >&2; exit 13\n",
    )
    world.write_config("termctl", hook_config("pre-open", (changed, None)))
    # The caller's cwd is the agent's own: from another directory this resume is refused (see
    # test_the_override_resume_from_another_cwd, an expected failure that records the divergence).
    resumed = world.spawn_json(
        "resume", eid, "--override-escalation", *HUMAN, timeout=120, cwd=wt.repo.clone
    )
    after = wait_status(world, eid, "live")["agent"]
    world.observe("agent-spawn.termctl-escalation.resumed", resumed["agent"]["status"])
    state = show(world, eid)
    pane = panes_of(world, eid)
    world.track(int(state["supervisor"]["pid"]), tree=True)
    check(
        marker.exists() and len(pane) == 1 and after["window"] is not None,
        "the human override reached termctl open: the changed hook ran again, its escalation "
        "counted as allow and the window exists",
        "interface.md Escalation overrides (resume: termctl -> termctl open --override-escalation); termctl Hooks (exit 13 from any pre-hook counts as allow)",
        (marker.exists(), pane, after["window"]),
    )
    last = json.loads(seen.read_text().strip().splitlines()[-1])
    check(
        last["override_escalation"] is True,
        "the pre-open payload of the overriding run has override_escalation true",
        "termctl interface.md Escalation override (every pre-hook payload of such a run says so)",
        last.get("override_escalation"),
    )
    check(
        after["needs_attention"] is None and after["escalated_by"] is None,
        "a live agent after the override has no attention and no escalated_by",
        "interface.md Agent object (escalated_by is non-null only while a hook escalation parked the agent)",
        {k: after[k] for k in ("needs_attention", "escalated_by")},
    )
    audit = world.state / "termctl" / "audit.jsonl"
    check(
        audit.exists() and any('"open"' in line for line in audit.read_text().splitlines()),
        "termctl wrote an audit record for the open run that carried --override-escalation",
        "termctl interface.md Audit and log (audit.jsonl: every open run with --override-escalation)",
        audit.read_text() if audit.exists() else "(no audit.jsonl)",
    )
    stop = world.spawn_json("stop", eid, "--grace", "2s", ok=(0,), timeout=60)
    _ = (os, signal, process_token, stop)


@SC("as-termctl-override-escalation")
@pytest.mark.xfail(
    strict=True,
    raises=LiveError,
    reason="product bug, reported with #112: resuming a launch whose pending run was canceled "
    "(termctl escalated before the session was adopted) runs claude-session start --resume in the "
    "caller's cwd, not the agent's recorded cwd; from a directory outside the repository "
    "claude-session refuses it (exit 12, end reason git_dir_changed)",
)
def test_the_override_resume_from_another_cwd(world: World) -> None:
    """interface.md Resume order step 4 (``Cwd: plain resume, the agent's recorded cwd``): the
    human override of a termctl escalation must work wherever the human runs ``resume``. It does
    not from a directory outside the repository (the harness's default cwd): the canceled pending
    run recorded no cwd, so claude-session resumes in the caller's cwd and refuses the changed git
    directory. strict: when the product is fixed this test starts passing and must be
    promoted to a plain test."""
    escalate = write_script(world, "pre-open-escalate.sh", "cat > /dev/null; exit 13\n")
    wt = stub_setup(world, "while :; do sleep 0.2; done")
    world.write_config("termctl", hook_config("pre-open", (escalate, None)))
    escalated = world.spawn_json(*window_launch_args(wt), ok=(13,), timeout=120)
    eid = escalated["error"]["agent_id"]
    world.write_config("termctl", hook_config("pre-open", (escalate, None)))
    world.spawn_json("resume", eid, "--override-escalation", *HUMAN, timeout=120)
    state = wait_status(world, eid, "live")
    world.track(int(state["supervisor"]["pid"]), tree=True)


def lock_file_of(world: World, socket_path: str) -> Path:
    """termctl's per-server lock file: ``locks/<first 16 hex digits of sha256(socket_path)>.lock``
    in its state directory (termctl interface.md, Audit and log, Locking)."""
    digest = hashlib.sha256(socket_path.encode()).hexdigest()[:16]
    return world.state / "termctl" / "locks" / f"{digest}.lock"


@contextmanager
def held_lock(path: Path) -> Iterator[None]:
    """Hold the exclusive ``flock`` termctl's label, rename, mark and unmark commands wait for, the
    way a concurrent termctl command would (the kernel makes no difference between holders)."""
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the flock


@SC("termctl-locked")
def test_termctl_locked_is_provoked_with_the_servers_lock_held(world: World) -> None:
    """termctl interface.md Audit and log (Locking: ``rename``, ``mark``, ``unmark``, ``label add``
    and ``label remove`` hold an exclusive ``flock`` on ``locks/<16 hex of sha256(socket_path)>.lock``
    and wait at most 10 s, then exit 12 ``locked`` with ``details.cause: timeout``; a lock file that
    cannot be used is ``cause: unusable`` with ``path`` and ``problem``; a dry run takes no lock;
    read-only commands take none). Provoked deterministically: the test holds the same ``flock``
    itself, so no second termctl has to be raced. #114, #36 #8 termctl (the ``locked`` error)."""
    opened = world.json_tool(
        "termctl", "open", "--name", "lk", "--label", "k=v", "--", "sh", "-c", "sleep 600"
    )
    pane = opened["id"]
    world.pids.append((int(opened["pid"]), process_token(int(opened["pid"])) or ""))
    socket_path = opened["socket_path"]
    # a first label change creates the lock file (and its directory)
    world.json_tool("termctl", "label", "add", pane, "a=b")
    lock = lock_file_of(world, socket_path)
    check(
        lock.exists(),
        "a label change leaves the server's lock file under the state directory's locks/",
        "termctl interface.md Audit and log (Locking: locks/<first 16 hex digits of sha256(socket_path)>.lock)",
        sorted(str(p) for p in (world.state / "termctl").rglob("*")),
    )

    with held_lock(lock):
        started = time.monotonic()
        refused = world.tool("termctl", "--json", "label", "add", pane, "c=d", timeout=60)
        waited = time.monotonic() - started
        world.observe("termctl.locked.timeout", (refused.returncode, round(waited, 1), refused.stdout))
        error = json.loads(refused.stdout)["error"]
        check(
            refused.returncode == 12
            and error["code"] == "locked"
            and error["details"]["cause"] == "timeout"
            and error["details"]["socket_path"] == socket_path,
            "label add with the server lock held exits 12 locked, details.cause timeout and the socket path",
            "termctl interface.md Errors (locked: details socket_path and cause), Audit and log (Locking)",
            error,
        )
        check(
            9.0 <= waited < 25.0,
            "termctl waited for the lock about 10 seconds before giving up",
            "termctl interface.md Audit and log (waits at most 10 seconds for the lock)",
            round(waited, 2),
        )
        labels = [p["labels"] for p in world.json_tool("termctl", "list")["panes"]]
        check(
            labels == [["a=b", "k=v"]] or labels == [["k=v", "a=b"]] or sorted(labels[0]) == ["a=b", "k=v"],
            "the refused label change changed nothing on the pane",
            "termctl interface.md Locking (the lock guards the read-modify-write)",
            labels,
        )
        # a dry run takes no lock and a read-only command takes none
        dry = world.tool("termctl", "--json", "label", "add", pane, "c=d", "--dry-run")
        check(
            dry.returncode == 0 and json.loads(dry.stdout)["dry_run"] is True,
            "a dry run never exits 12: it takes no lock",
            "termctl interface.md Dry runs (a dry run takes no lock)",
            (dry.returncode, dry.stdout[-300:]),
        )
        listed = world.tool("termctl", "--json", "list")
        check(listed.returncode == 0, "list takes no lock", "termctl interface.md Locking", listed.stderr)

    # released: the same command succeeds (the lock is not left behind by the refused one)
    ok = world.json_tool("termctl", "label", "add", pane, "c=d")
    check(
        ok["changed"] is True and "c=d" in ok["labels"],
        "after the holder let go the same label change succeeds",
        "termctl interface.md Locking (the kernel releases a flock with its holder)",
        ok,
    )

    # a lock file that cannot be used: cause unusable with path and problem (no wait); a root
    # user opens a mode-000 file anyway, so the harness only provokes it for other users
    if os.geteuid() == 0:
        world.json_tool("termctl", "close", pane, "--force", "--actor", "human:tester")
        return
    lock.chmod(0)
    try:
        started = time.monotonic()
        unusable = world.tool("termctl", "--json", "label", "add", pane, "e=f", timeout=60)
        quick = time.monotonic() - started
        world.observe("termctl.locked.unusable", (unusable.returncode, unusable.stdout))
        error = json.loads(unusable.stdout)["error"]
        check(
            unusable.returncode == 12
            and error["code"] == "locked"
            and error["details"]["cause"] == "unusable"
            and error["details"]["path"].endswith(lock.name)
            and error["details"]["problem"],
            "an unreadable lock file exits 12 locked with details.cause unusable, a path and a problem",
            "termctl interface.md Errors (locked details: cause unusable, path, problem)",
            error,
        )
        check(
            quick < 8.0,
            "an unusable lock fails at once instead of waiting 10 seconds",
            "termctl interface.md Locking (unusable is not a timeout)",
            round(quick, 2),
        )
    finally:
        lock.chmod(0o600)
    world.json_tool("termctl", "close", pane, "--force", "--actor", "human:tester")


@SC("as-label-locked")
def test_agent_spawn_label_add_reports_a_locked_pane_and_resyncs_later(world: World) -> None:
    """interface.md ``label add`` (a change is mirrored to the pane of a live window with
    ``termctl label add``; a mirror failure is a warning and queues the change for re-sync) and
    Reconcile, label re-sync (``labels_resynced``): with termctl's server lock held, the real
    termctl answers ``locked`` after its 10 s wait, agent-spawn keeps the change on the agent and
    the session, warns, and a later ``reconcile`` pushes it once the lock is free. #114: how
    agent-spawn reports ``locked``."""
    wt = stub_setup(world, "while :; do sleep 0.2; done")
    launched = world.spawn_json(*window_launch_args(wt), timeout=120)
    aid = agent_of(launched)["agent_id"]
    state = wait_status(world, aid, "live")
    world.track(int(state["supervisor"]["pid"]), tree=True)
    pane = panes_of(world, aid)[0]
    lock = lock_file_of(world, pane["socket_path"])
    world.json_tool("termctl", "label", "add", pane["id"], "warm=up")  # creates the lock file
    with held_lock(lock):
        changed = world.spawn_json("label", "add", aid, "team=blue", timeout=90)
    world.observe("agent-spawn.label-add.locked", changed)
    check(
        changed["changed"] is True
        and "team=blue" in changed["labels"]
        and any("termctl label add failed (exit 12)" in w and "queued" in w for w in changed["warnings"]),
        "label add with the pane's server lock held succeeds on the agent and warns that termctl's "
        "label add failed with exit 12 (locked) and the change is queued",
        "interface.md label add (a mirror failure is a warning and queues the change); termctl Locking",
        changed,
    )
    csid = show(world, aid)["agent"]["csession_id"]
    check(
        "team=blue" in world.json_tool("claude-session", "status", csid)["labels"],
        "the session mirror took the change although the pane push failed",
        "interface.md label add (mirrored to the session)",
    )
    check(
        "team=blue" not in panes_of(world, aid)[0]["labels"],
        "the pane does not carry the label yet (termctl refused it)",
        "termctl interface.md Locking (a locked command changes nothing)",
        panes_of(world, aid)[0]["labels"],
    )
    synced = world.spawn_json("reconcile", ok=(0, 12))
    kinds = [c["kind"] for c in synced["changes"]]
    world.observe("agent-spawn.reconcile.labels", synced)
    check(
        "labels_resynced" in kinds and "team=blue" in panes_of(world, aid)[0]["labels"],
        "a reconcile after the lock is free pushes the queued label to the pane",
        "interface.md Reconcile (labels_resynced, Label re-sync scope)",
        (kinds, panes_of(world, aid)[0]["labels"]),
    )
    world.spawn_json("stop", aid, "--grace", "2s", timeout=60)
