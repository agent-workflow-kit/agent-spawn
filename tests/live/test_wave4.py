"""The unbilled live follow-ups of wave 4 (#112 to #116): hooks, termctl, wtctl and claude-session
behaviors the earlier waves left out, against the real tools with a stub ``claude`` (no Claude
session, no credentials, no billing).

Human-only flags are driven the way the harness drives every human-only gate: the children have no
TTY on stdin, so the tests pass an explicit ``--actor human:<user>`` (the one TTY case that a pty
can provide is its own scenario, ``tty-human-gates``).
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pty
import re
import signal
import sqlite3
import subprocess
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from .billed import pid_gone, signal_exact
from .harness import (
    MINIMUMS,
    LiveError,
    World,
    check,
    parse_version,
    process_token,
    wait_for,
    write_fake_crontab,
)
from .test_agent_spawn_hooks import HUMAN, hook_config, launch_args, write_script
from .test_agent_spawn_stub import agent_of, show, stub_setup, wait_status
from .test_claude_session import cs, cs_raw, prepare, start_args
from .test_wtctl import AUTO, Wt
from .test_wtctl_hooks import create_args, err_of, wt_config

SC = pytest.mark.live_scenario

#: ``claude-session wait`` of a run that ended by ``stop``: observed with claude-session 0.45.2.
WAIT_AFTER_STOP_EXIT = 12


class KnownDivergence(LiveError):
    """Raised at the single point where a known, filed product or tool divergence shows. The strict
    expected failures name it in ``raises=``, so any other failure of the same test is a real one."""


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
    hook_pid = world.root / "hook.pid"
    # the pids are written first, so a slow machine cannot end the hook before they exist
    slow = write_script(
        world,
        "slow-teardown.sh",
        f"echo $$ > {hook_pid}\nsleep 120 &\necho $! > {sleeper}\necho slow >> {ran}\n"
        "cat > /dev/null\nwait\n",
    )
    wt = stub_setup(world, "exit 0", agent_spawn_config=hook_config("pre-teardown", (slow, 3)))

    def pid_of(path: Path) -> int:
        text = wait_for(
            lambda: path.exists() and path.read_text().strip(), 20, f"{path.name} to be written"
        )
        return int(text)

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
    for path in (hook_pid, sleeper):
        pid = pid_of(path)
        wait_for(
            lambda pid=pid: pid_gone(pid), 20, f"the timed-out hook's {path.name} process to end"
        )
    after = show(world, aid)["agent"]
    check(
        after["status"] == "parked" and after["teardown_step"] is None,
        "the vetoed teardown returned the agent to parked with no teardown marker",
        "interface.md Teardown order step 3 (either returns the agent to parked unchanged)",
        {k: after[k] for k in ("status", "teardown_step")},
    )
    check(
        after["csession_id"] == before["csession_id"]
        and after["needs_attention"] == before["needs_attention"],
        "nothing after the hook ran: the agent's session and attention are unchanged "
        "(the launch uses --worktree none, so no worktree is checked)",
        "interface.md Teardown order (the hook is step 3, before any removal)",
        {"before": before, "after": after},
    )
    check(
        len(world.json_tool("claude-session", "list")["sessions"]) == 1,
        "the session was not forgotten by the vetoed teardown",
        "interface.md Teardown order step 6 (forget happens after the hooks)",
    )

    # a human override lifts an escalation only: a timeout still vetoes, and the hook ran again
    # (the first run's pid files are removed so pid_of waits for the second run's own pids)
    hook_pid.unlink()
    sleeper.unlink()
    again = world.spawn_json("teardown", aid, "--override-escalation", *HUMAN, ok=(11,), timeout=90)
    check(
        again["error"]["code"] == "vetoed" and ran.read_text().count("slow") == 2,
        "--override-escalation does not lift a timeout veto: the hook ran again and vetoed again",
        "interface.md Escalation overrides (a veto still vetoes)",
        (again["error"]["code"], ran.read_text()),
    )
    for path in (hook_pid, sleeper):
        wait_for(
            lambda path=path: pid_gone(pid_of(path)),
            20,
            f"the second run's {path.name} process to end",
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
        world,
        "pre-open-escalate.sh",
        f"cat >> {seen}; echo >> {seen}; echo 'needs a human' >&2; exit 13\n",
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
    world.spawn_json("stop", eid, "--grace", "2s", ok=(0,), timeout=60)


@SC("as-termctl-override-escalation")
@pytest.mark.xfail(
    strict=True,
    raises=KnownDivergence,
    reason="product bug agent-spawn#119: resuming a launch whose pending run was canceled "
    "(termctl escalated before the session was adopted) runs claude-session start --resume in the "
    "caller's cwd, not the agent's recorded cwd; from a directory outside the repository "
    "claude-session refuses it (exit 12, end reason git_dir_changed)",
)
def test_the_override_resume_from_another_cwd(world: World) -> None:
    """interface.md Resume order step 4 (``Cwd: plain resume, the agent's recorded cwd``): the
    human override of a termctl escalation must work wherever the human runs ``resume``. It does
    not from a directory outside the repository (the harness's default cwd): the canceled pending
    run recorded no cwd, so claude-session resumes in the caller's cwd and refuses the changed git
    directory (agent-spawn#119). The test raises ``KnownDivergence`` only at that point (the
    escalated launch exited 13, the resume exited 12 with ``git_dir_changed``); anything else is a
    real failure. strict: when the product is fixed this test starts passing and must be
    promoted to a plain test."""
    escalate = write_script(world, "pre-open-escalate.sh", "cat > /dev/null; exit 13\n")
    wt = stub_setup(world, "while :; do sleep 0.2; done")
    world.write_config("termctl", hook_config("pre-open", (escalate, None)))
    escalated = world.spawn_json(*window_launch_args(wt), ok=(13,), timeout=120)
    eid = escalated["error"]["agent_id"]
    resumed = world.agent_spawn(
        "resume", eid, "--override-escalation", *HUMAN, "--json", timeout=120
    )
    world.observe(
        "agent-spawn.termctl-escalation.resume-other-cwd",
        (resumed.returncode, resumed.stdout[-600:]),
    )
    if resumed.returncode == 12 and "git_dir_changed" in resumed.stdout + resumed.stderr:
        raise KnownDivergence(
            "resume --override-escalation from another cwd exits 12 git_dir_changed (agent-spawn#119): "
            + resumed.stdout[-300:]
        )
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
        world.observe(
            "termctl.locked.timeout", (refused.returncode, round(waited, 1), refused.stdout)
        )
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
            len(labels) == 1 and sorted(labels[0]) == ["a=b", "k=v"],
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
        check(
            listed.returncode == 0,
            "list takes no lock",
            "termctl interface.md Locking",
            listed.stderr,
        )

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
        and any(
            "termctl label add failed (exit 12)" in w and "queued" in w for w in changed["warnings"]
        ),
        "label add with the pane's server lock held succeeds on the agent and warns "
        "'termctl label add failed (exit 12)' and that the change is queued",
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


def wtctl_db_update(world: World, sql: str, *params: str) -> None:
    """Edit wtctl's private ``state.db`` (the harness world's own): the only way to present the
    states wtctl's own docs describe but a test cannot make with git alone (a digest that
    disagrees with the recorded rows; a file system that presents a new ``st_dev``)."""
    found = list((world.state / "wtctl").rglob("state.db"))
    check(len(found) == 1, "the world has exactly one wtctl state.db", "harness isolation", found)
    db = sqlite3.connect(str(found[0]))
    try:
        db.execute(sql, params)
        db.commit()
    finally:
        db.close()


@SC("wtctl-baseline-no-classes")
def test_a_digest_only_and_an_anchor_only_baseline_change(world: World) -> None:
    """wtctl interface.md Baseline check (a change is exit 20 ``repo_config_changed`` and the error
    carries ``items`` and ``details.changed_classes``), Baseline accept (a moved anchor is shown
    with the changed items; ``--expect-digest``) and Integrity (``unlinked``). Observed here, and
    the agent-spawn side is ``as-wtctl-baseline-no-classes``: a recorded digest that disagrees with
    the recorded rows is ``repo_config_changed`` with only ``digest`` and ``recorded_digest`` (no
    ``items``, no ``changed_classes``); a moved anchor identity is NOT a baseline change for
    ``baseline check`` (integrity refuses it first: ``unlinked``) and surfaces as
    ``repo_config_changed`` with ``details.anchors`` and no ``changed_classes`` only from
    ``baseline accept --expect-digest``. #115, #36 #25."""
    wt = Wt(world)
    made = wt.create("nb1", "ag-aaaa")["record"]
    rid = made["id"]
    clean = wt.run("baseline", "check", rid)
    check(clean.returncode == 0, "a fresh record's baseline is unchanged", "#36 #25", clean.stdout)

    # digest only
    wtctl_db_update(world, "update records set baseline_digest = ? where id = ?", "0" * 64, rid)
    done = wt.run("baseline", "check", rid)
    world.observe("wtctl.baseline.digest-only", (done.returncode, done.stdout))
    error = json.loads(done.stdout)["error"]
    check(
        done.returncode == 20
        and error["code"] == "refused"
        and error["reason"] == "repo_config_changed"
        and sorted(error["details"]) == ["digest", "recorded_digest"]
        and "items" not in error
        and error["details"]["recorded_digest"] == "0" * 64,
        "a digest that disagrees with the recorded rows is refused repo_config_changed with only the two digests: no items and no changed_classes",
        "wtctl interface.md Baseline check (exit 20, reason repo_config_changed; items and details only when rows differ)",
        error,
    )
    status = wt.run("status", rid)
    world.observe("wtctl.baseline.digest-only.status", (status.returncode, status.stdout[:600]))
    check(
        json.loads(status.stdout)["reason"] == "repo_config_changed"
        and json.loads(status.stdout)["would_remove"] is False,
        "status treats the digest-only change as a baseline refusal: remove would refuse repo_config_changed",
        "wtctl interface.md Baseline (remove refuses repo_config_changed)",
        status.stdout[:400],
    )
    accepted = wt.run("baseline", "accept", rid, "--yes", "--actor", "human:tester")
    check(
        accepted.returncode == 0
        and wt.run("baseline", "check", rid).returncode == 0
        and json.loads(accepted.stdout)["items"] == [],
        "a human baseline accept re-records the digest (no items to show) and the check is clean again",
        "wtctl interface.md baseline accept",
        accepted.stdout[:300],
    )

    # anchor only: dir_id, then the clone's common_id
    for column, anchor in (("dir_id", "worktree_dir"), ("common_id", "common_dir")):
        wtctl_db_update(world, f"update records set {column} = ? where id = ?", "1:2", rid)
        done = wt.run("baseline", "check", rid)
        world.observe(f"wtctl.baseline.anchor-only.{column}.check", (done.returncode, done.stdout))
        error = json.loads(done.stdout)["error"]
        check(
            done.returncode == 20 and error["reason"] == "unlinked",
            f"a moved {column} is refused unlinked by baseline check (integrity runs first), not repo_config_changed",
            "wtctl interface.md Baseline check (integrity first: unlinked), Mount points",
            error,
        )
        refused = wt.run(
            "baseline",
            "accept",
            rid,
            "--yes",
            "--actor",
            "human:tester",
            "--expect-digest",
            "f" * 64,
        )
        world.observe(
            f"wtctl.baseline.anchor-only.{column}.accept", (refused.returncode, refused.stdout)
        )
        error = json.loads(refused.stdout)["error"]
        details = error["details"]
        check(
            refused.returncode == 20
            and error["reason"] == "repo_config_changed"
            and "changed_classes" not in details
            and "items" not in error
            and [a["anchor"] for a in details["anchors"] if a["changed"]] == [anchor]
            and details["digest"] == details["recorded_digest"],
            f"baseline accept --expect-digest with only {anchor} moved: repo_config_changed with details.anchors, no items and no changed_classes",
            "wtctl interface.md baseline accept (a moved anchor is shown with the changed items; the digest is rechecked)",
            error,
        )
        ok = wt.run("baseline", "accept", rid, "--yes", "--actor", "human:tester")
        check(
            ok.returncode == 0 and wt.run("baseline", "check", rid).returncode == 0,
            f"a human accept re-records the moved {column} and the check is clean again",
            "wtctl interface.md baseline accept (re-recorded with them)",
            ok.stdout[:300],
        )


@SC("as-wtctl-baseline-no-classes")
def test_agent_spawn_show_and_teardown_for_a_baseline_change_without_classes(world: World) -> None:
    """interface.md ``show`` (``repo_config_changed`` from ``wtctl baseline check`` of the agent's
    own record, for information) and Teardown order step 4 (wtctl's refusal is passed through):
    for a digest-only change (``changed_classes`` absent: the wrapper maps it to ``changed: true``
    with no classes and no items) and for a moved anchor (``baseline check`` says ``unlinked``,
    so ``show`` reports no ``repo_config_changed`` and warns that the baseline was not checked).
    The teardown of both is refused by wtctl and passes through; after a human ``wtctl baseline
    accept`` the teardown goes through. #115."""
    wt = stub_setup(world, "exit 0")
    launched = world.spawn_json(*new_worktree_launch(wt), ok=(14,))
    aid = agent_of(launched)["agent_id"]
    rid = agent_of(launched)["worktree"]["record_id"]
    check(
        show(world, aid)["repo_config_changed"]["changed"] is False,
        "an untouched worktree shows repo_config_changed.changed false",
        "interface.md show (repo_config_changed)",
    )

    # digest only
    wtctl_db_update(world, "update records set baseline_digest = ? where id = ?", "0" * 64, rid)
    shown = show(world, aid)
    world.observe("agent-spawn.show.digest-only", shown["repo_config_changed"])
    check(
        shown["repo_config_changed"] == {"changed": True, "changed_classes": [], "items": []},
        "a digest-only change shows repo_config_changed changed true with no classes and no items",
        "interface.md show (repo_config_changed: {changed, changed_classes, items})",
        shown["repo_config_changed"],
    )
    refused = world.spawn_json("teardown", aid, ok=(20,))
    world.observe("agent-spawn.teardown.digest-only", refused["error"])
    check(
        refused["error"]["tool"] == "wtctl"
        and refused["error"]["lower_error"]["reason"] == "repo_config_changed",
        "teardown passes wtctl's repo_config_changed refusal of the digest-only change through (exit 20)",
        "interface.md Teardown order step 4 (wtctl refusals are passed through)",
        refused["error"],
    )
    check(
        show(world, aid)["agent"]["status"] == "parked"
        and Path(agent_of(launched)["worktree"]["path"]).is_dir(),
        "the refused teardown left the agent parked and the worktree in place",
        "interface.md Teardown order (a refusal that changed nothing returns the agent to parked)",
    )
    ok = wt.run("baseline", "accept", rid, "--yes", "--actor", "human:tester")
    check(
        ok.returncode == 0,
        "the human accept clears the digest-only change",
        "wtctl baseline accept",
        ok.stdout[:300],
    )

    # a moved anchor
    wtctl_db_update(world, "update records set dir_id = ? where id = ?", "1:2", rid)
    shown = show(world, aid)
    world.observe("agent-spawn.show.anchor-only", (shown["repo_config_changed"], shown["warnings"]))
    check(
        shown["repo_config_changed"] is None
        and any("baseline was not checked" in w for w in shown["warnings"]),
        "a moved anchor makes baseline check refuse unlinked: show has no repo_config_changed and warns that it was not checked",
        "interface.md show (for information); wtctl Baseline check (integrity first)",
        (shown["repo_config_changed"], shown["warnings"]),
    )
    refused = world.spawn_json("teardown", aid, ok=(20,))
    world.observe("agent-spawn.teardown.anchor-only", refused["error"])
    check(
        refused["error"]["tool"] == "wtctl"
        and refused["error"]["lower_error"]["reason"] == "unlinked",
        "teardown passes wtctl's unlinked refusal of the moved anchor through (exit 20)",
        "interface.md Teardown order step 4 (wtctl refusals are passed through)",
        refused["error"],
    )
    ok = wt.run("baseline", "accept", rid, "--yes", "--actor", "human:tester")
    check(
        ok.returncode == 0,
        "the human accept re-records the anchor",
        "wtctl baseline accept",
        ok.stdout[:300],
    )
    torn = world.spawn_json("teardown", aid)
    check(
        [w["action"] for w in torn["worktrees"]] == ["removed"],
        "after the accept the teardown removes the worktree",
        "interface.md Teardown order step 4",
        torn["worktrees"],
    )


def new_worktree_launch(wt: Any) -> list[str]:
    return [
        "launch", "--mode", "headless", "--prompt", "reply ok", "--model", "haiku",
        "--repo", wt.repo.repo_id, "--worktree", "new", "--wait",
    ]  # fmt: skip


class Background:
    """A child the test starts and waits for later (a ``wtctl create`` held at a barrier). Its pid
    is tracked so the world's teardown ends it (exact pid plus start token) even if a check
    fails; output goes to files, never to a pipe nobody reads."""

    def __init__(self, world: World, argv: Sequence[str]) -> None:
        self.out = world.root / f"bg-{len(list(world.root.glob('bg-*.out')))}.out"
        self.err = self.out.with_suffix(".err")
        with self.out.open("w") as out, self.err.open("w") as err:
            self.child = subprocess.Popen(
                list(argv),
                env=world.env(),
                cwd=world.root / "repos",
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        world.track(self.child.pid)

    def kill(self) -> None:
        """End the group this class created (the child is its leader), if it still runs."""
        if self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGKILL)
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def finish(self, timeout: float = 60) -> tuple[int, str]:
        try:
            code = self.child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            raise LiveError(
                f"the background command did not finish within {timeout:.0f}s"
            ) from None
        return code, self.out.read_text()


def barrier_script(world: World, name: str, started: Path, go: Path, then: str = "") -> Path:
    """A hook or bootstrap step that announces itself and waits (bounded, 60 s) for ``go``."""
    return write_script(
        world,
        name,
        f"cat > /dev/null 2>&1 < /dev/null\necho x >> {started}\ni=0\n"
        f"while [ ! -e {go} ] && [ $i -lt 600 ]; do sleep 0.1; i=$((i+1)); done\n{then}",
    )


@SC("wtctl-create-busy")
def test_a_concurrent_create_and_a_command_on_a_creating_record_are_busy(world: World) -> None:
    """wtctl interface.md Create order (step 4: the per-clone lease is held through step 13, so a
    concurrent ``create`` exits 12 ``busy`` having allocated no record id; step 12: the record is
    ``creating`` under its record lease through step 16, and the clone lease is released before
    bootstrap) and Locking (a command waits ``lease_wait`` for a held lease, then 12 ``busy``),
    States (``A live creator``: ``remove`` and ``forget`` on a ``creating`` record wait, then
    ``busy``). The creator is a real ``wtctl create`` held at a barrier: first inside a
    ``pre-create`` hook (clone lease held), then inside a bootstrap step (record ``creating``,
    clone lease released). #116 (``create`` ``busy``, the ``creating`` state)."""
    wt = Wt(world)
    hook_started, hook_go = world.root / "hook.started", world.root / "hook.go"
    step_started, step_go = world.root / "step.started", world.root / "step.go"
    hook = barrier_script(world, "barrier-hook.sh", hook_started, hook_go)
    step = barrier_script(world, "barrier-step.sh", step_started, step_go)
    wt_config(world, wt, hooks={"pre-create": [(hook, None)]}, steps=[step])
    config = world.config_home / "wtctl" / "config.yaml"
    config.write_text(config.read_text() + "lease_wait: 2s\n")

    first = Background(
        world,
        [str(world.bin / "wtctl"), "--json", "create", "--repo", wt.repo.repo_id, "--name", "c1",
         "--holder", "ag-c1", "--actor", "auto:agent-spawn"],
    )  # fmt: skip
    try:
        wait_for(hook_started.exists, 30, "the first create to reach its pre-create hook")
        check(
            wt.ok("list")["records"] == [],
            "no record exists while the creator is still in its pre-create hook (step 11 precedes step 12)",
            "wtctl interface.md Create order steps 11, 12",
        )

        # 1. a concurrent create waits lease_wait for the per-clone lease and exits 12 busy
        started = time.monotonic()
        busy = wt.run(*create_args(wt, "c2"), f"--actor={AUTO}", timeout=60)
        waited = time.monotonic() - started
        world.observe("wtctl.create.busy", (busy.returncode, round(waited, 1), busy.stdout))
        check(
            busy.returncode == 12 and err_of(busy)["code"] == "busy",
            "a create while another create holds the clone lease exits 12 with error.code busy",
            "wtctl interface.md Create order step 4, Exit codes (busy is 12)",
            (busy.returncode, busy.stdout[-300:]),
        )
        check(
            2.0 <= waited < 20.0,
            "the busy create waited lease_wait (2 s) first",
            "wtctl interface.md Locking (a command waits up to lease_wait)",
            round(waited, 2),
        )
        check(
            hook_started.read_text().count("x") == 1 and wt.ok("list")["records"] == [],
            "the busy create ran no hook and left no record",
            "wtctl interface.md Create order step 4 (busy allocated no record id)",
        )

        # 2. let the creator on: it is now in its bootstrap step, a creating record
        hook_go.write_text("go")
        wait_for(step_started.exists, 30, "the first create to reach its bootstrap step")
        records = wt.ok("list")["records"]
        world.observe("wtctl.creating.record", records)
        check(
            [r["state"] for r in records] == ["creating"] and records[0]["holders"] == ["ag-c1"],
            "while its bootstrap runs the record is creating and holds the creator's holder",
            "wtctl interface.md Create order step 12, Record states (creating)",
            records,
        )
        rid = records[0]["id"]
        shown = wt.ok("show", rid)
        check(
            (shown.get("record") or shown)["state"] == "creating",
            "show reports the creating state too",
            "wtctl interface.md Record states",
            shown,
        )
        # the clone lease is free again: another create goes through (without bootstrap)
        other = wt.ok(*create_args(wt, "c3"), "--no-bootstrap", f"--actor={AUTO}")
        check(
            other["record"]["state"] == "active",
            "a create without bootstrap succeeds while the first create is bootstrapping (the clone lease was released before bootstrap)",
            "wtctl interface.md Create order step 4 (held through step 13, released before bootstrap)",
            other["record"],
        )
        # commands on the creating record wait for its lease and then are busy
        for args in (
            ("remove", rid, "--yes", "--actor", "human:tester"),
            ("forget", rid, "--actor", "human:tester"),
        ):
            started = time.monotonic()
            held = wt.run(*args, timeout=60)
            waited = time.monotonic() - started
            check(
                held.returncode == 12 and err_of(held)["code"] == "busy" and waited >= 2.0,
                f"{args[0]} of a creating record whose creator is alive waits lease_wait then exits 12 busy",
                "wtctl interface.md Record states (A live creator), Locking",
                (held.returncode, round(waited, 1), held.stdout[-300:]),
            )
        still = [r for r in wt.ok("list")["records"] if r["id"] == rid]
        check(
            len(still) == 1
            and still[0]["state"] == "creating"
            and still[0]["holders"] == ["ag-c1"],
            "the busy remove and forget changed nothing: the record is still creating with the creator's holder",
            "wtctl interface.md Locking, Record states (A live creator)",
            still,
        )
    except BaseException:
        first.kill()
        raise
    finally:
        hook_go.write_text("go")
        step_go.write_text("go")
    code, out = first.finish(timeout=60)
    created = json.loads(out)
    check(
        code == 0 and created["record"]["state"] == "active" and created["record"]["id"] == rid,
        "the held create finishes normally once released: the same record, now active",
        "wtctl interface.md Create order steps 14-16",
        (code, out[-300:]),
    )
    later = wt.run(*create_args(wt, "c2"), f"--actor={AUTO}", "--no-bootstrap")
    check(
        later.returncode == 0,
        "the create that was busy succeeds once the lease is free (no stale lease)",
        "wtctl interface.md Locking",
        later.stdout[-300:],
    )


@SC("as-teardown-partial")
def test_a_remove_that_leaves_an_entry_is_partial_and_teardown_stays_sticky(world: World) -> None:
    """wtctl interface.md Remove order step 9 (Failure: an entry that cannot be deleted leaves the
    record ``partial`` and the command exits 1, ``error.code: partial``, ``details.remaining_path``)
    and agent-spawn interface.md Teardown order step 4 (partial and unknown outcomes are sticky:
    wtctl exit 1 ``partial`` is exit 12 ``partial``, the agent stays ``tearing_down`` at
    ``worktree_partial``, a retry over a ``partial`` record stays there, ``reconcile`` reports
    ``teardown_incomplete``, a human's ``wtctl remove`` finishes the record and the retry
    completes). The failure is made without git or root: the worktree's parent directory loses its
    write bit, so wtctl deletes the tree but cannot remove the directory itself. #116 (``remove``
    exit 1 ``partial``); the killed-mid-delete half is not covered (see docs/live-testing.md)."""
    if os.geteuid() == 0:
        pytest.skip(
            "root ignores a directory's missing write bit: the partial remove cannot be provoked"
        )
    wt = stub_setup(world, "exit 0")
    launched = world.spawn_json(*new_worktree_launch(wt), ok=(14,))
    aid = agent_of(launched)["agent_id"]
    record = agent_of(launched)["worktree"]
    path = Path(record["path"])
    parent = path.parent
    check(
        parent.resolve().is_relative_to(world.root.resolve()),
        "the directory to make unwritable lies inside the private world",
        "docs/live-testing.md Isolation",
        parent,
    )
    parent.chmod(0o555)
    try:
        first = world.spawn_json("teardown", aid, ok=(12,))
        world.observe("agent-spawn.teardown.partial", first["error"])
        error = first["error"]
        check(
            error["code"] == "partial"
            and error["tool"] == "wtctl"
            and error["lower_code"] == "partial"
            and error["lower_error"]["details"]["remaining_path"] == str(path),
            "wtctl exit 1 partial is passed through as agent-spawn exit 12 partial, tool wtctl, naming the remaining path",
            "interface.md Teardown order step 4 (Partial and unknown outcomes are sticky)",
            error,
        )
        stuck = show(world, aid)["agent"]
        check(
            stuck["status"] == "tearing_down" and stuck["teardown_step"] == "worktree_partial",
            "the agent stays tearing_down with the marker worktree_partial",
            "interface.md Teardown order step 4; Lifecycle (never back to parked over such a record)",
            {k: stuck[k] for k in ("status", "teardown_step")},
        )
        check(
            wt.ok("show", record["record_id"])["state"] == "partial",
            "wtctl shows the record partial",
            "wtctl interface.md Remove order step 9 (Failure)",
        )
        # a retry over the still-partial record stays put (it never calls remove again)
        second = world.spawn_json("teardown", aid, ok=(12,))
        check(
            second["error"]["code"] == "partial",
            "a retry while the record is still partial exits 12 partial again",
            "interface.md Teardown order step 4 (state partial -> the agent stays tearing_down, exit 12 partial)",
            second["error"],
        )
        reconciled = world.spawn_json("reconcile", ok=(0, 12))
        incomplete = [c for c in reconciled["changes"] if c["kind"] == "teardown_incomplete"]
        world.observe("agent-spawn.reconcile.teardown-incomplete", incomplete)
        check(
            len(incomplete) == 1
            and incomplete[0]["details"]["teardown_step"] == "worktree_partial"
            and incomplete[0]["details"]["partial_record"] == record["record_id"],
            "reconcile reports teardown_incomplete with the worktree_partial step and the record",
            "interface.md Reconcile (teardown_incomplete has details.teardown_step and details.partial_record)",
            reconciled["changes"],
        )
    finally:
        parent.chmod(0o755)
    # a human finishes the partial record; the retried teardown then completes
    finished = wt.run("remove", record["record_id"], "--yes", "--actor", "human:tester")
    check(
        finished.returncode == 0,
        "a human wtctl remove finishes the partial record",
        "wtctl interface.md Partial records (human-only)",
        (finished.returncode, finished.stdout[-300:]),
    )
    torn = world.spawn_json("teardown", aid)
    check(
        show(world, aid)["agent"]["status"] == "torn_down" and torn["session_forgotten"] is True,
        "after the human finished the record the retried teardown completes: torn_down",
        "interface.md Teardown order step 4 (a tombstone completes the record and teardown continues)",
        torn,
    )


@SC("as-input-files")
def test_input_files_are_copied_staged_carried_and_refused_by_claude_session(world: World) -> None:
    """claude-session interface.md Launch flags (``--input-file``: copied to
    ``$CSESSION_DIR/inputs/<run>/<basename>``, readable (mode 0600 observed), unique basenames from the
    artifact-name grammar, ``-`` rejected, each at most ``input_max_bytes``) and Prompts and inputs
    (a resume given its own inputs replaces the carried set; earlier inputs stay readable);
    agent-spawn interface.md Launch order step 1 (``--input-file`` is passed as a private copy that
    is deleted once a session is allocated). The stub claude lists what it can read under
    ``$CSESSION_DIR/inputs`` with the mode and content of each file. #116 (``--input-file``
    copies, needing an input fixture)."""
    seen = world.root / "inputs.seen"
    wt = stub_setup(
        world,
        f'for f in "$CSESSION_DIR"/inputs/*/*; do echo "$f $(stat -c %a "$f") $(cat "$f")" >> {seen}; done; exit 0',
    )
    fixtures = world.root / "fixtures"
    fixtures.mkdir()
    note, other = fixtures / "note.txt", fixtures / "other.md"
    note.write_text("first input\n")
    other.write_text("second input\n")

    launched = world.spawn_json(
        *launch_args(wt, "--wait", "--input-file", str(note)), ok=(14,), timeout=120
    )
    aid, csid = agent_of(launched)["agent_id"], agent_of(launched)["csession_id"]
    world.note_session(csid)
    note.write_text("changed after launch\n")  # the staged copy is independent of the caller's file
    lines = seen.read_text().splitlines()
    world.observe("agent-spawn.input-file.seen", lines)
    check(
        len(lines) == 1 and len(lines[0].split()) == 4,
        "the stub listed exactly one input: path, mode and the two words of its content",
        "claude-session interface.md Launch flags (--input-file)",
        lines,
    )
    check(
        lines[0].endswith("/inputs/1/note.txt " + lines[0].split()[1] + " first input")
        and int(lines[0].split()[1], 8) & 0o400 != 0
        and int(lines[0].split()[1], 8) & 0o077 == 0,
        "the run could read inputs/1/note.txt with the original content from an owner-only file "
        "(mode 0600; the sandbox's write deny for isolated runs is not checked here: the host has no bwrap)",
        "claude-session interface.md Launch flags (--input-file: copied, readable)",
        lines,
    )
    status = world.json_tool("claude-session", "status", csid)
    check(
        status["run"]["input_files"] == ["note.txt"],
        "status lists the run's input files by basename",
        "claude-session interface.md status (input_files)",
        status["run"]["input_files"],
    )
    prompts = world.state / "agent-spawn" / "prompts" / aid
    check(
        not prompts.exists() or not any(prompts.iterdir()),
        "agent-spawn's private copy of the input was deleted once the session was allocated",
        "interface.md Launch order step 1 (copies deleted once a session is allocated)",
        list(prompts.iterdir()) if prompts.exists() else [],
    )

    # a resume given its own input set replaces the carried one; the earlier input stays readable
    seen.write_text("")
    world.spawn_json(
        "resume", aid, "--mode", "headless", "--prompt", "again", "--input-file", str(other),
        "--wait", ok=(14,), timeout=120,
    )  # fmt: skip
    names = sorted(line.split()[0].split("/inputs/")[1] for line in seen.read_text().splitlines())
    check(
        names == ["1/note.txt", "2/other.md"],
        "after a resume with its own input the new file is in inputs/2 and the earlier one stays readable in inputs/1",
        "claude-session interface.md Prompts and inputs (earlier inputs stay readable in inputs/<n>/)",
        names,
    )
    check(
        world.json_tool("claude-session", "status", csid)["run"]["input_files"] == ["other.md"],
        "the second run's input set is the one given to the resume",
        "claude-session interface.md Carry rules (given now: replaces the carried input set)",
    )

    # refusals: exit 2, nothing allocated (direct claude-session)
    work = prepare(world, "lost_grace: 600s\ninput_max_bytes: 16\n")
    # the default prompt file is larger than 16 bytes: shrink it, or every start would be refused
    # for the prompt and the refusals below would prove nothing about --input-file
    (world.root / "prompt.md").write_text("ok\n")
    sessions = len(world.json_tool("claude-session", "list")["sessions"])
    big = fixtures / "big.txt"
    big.write_text("x" * 17)
    nested = fixtures / "sub"
    nested.mkdir()
    (nested / "note.txt").write_text("dup\n")
    bad_name = fixtures / "bad name.txt"
    bad_name.write_text("x\n")
    for what, extra, keyword in (
        ("an input over input_max_bytes", ("--input-file", str(big)), "input_max_bytes"),
        (
            "duplicate input basenames",
            ("--input-file", str(note), "--input-file", str(nested / "note.txt")),
            "duplicate",
        ),
        (
            "an input whose basename is outside the artifact-name grammar",
            ("--input-file", str(bad_name)),
            "must match",
        ),
        ("--input-file -", ("--input-file", "-"), "not supported"),
    ):
        done = cs_raw(world, *start_args(world, work, "agent=ag-input", *extra))
        world.observe(
            f"claude-session.input-file.refused.{what}", (done.returncode, done.stdout[-200:])
        )
        check(
            done.returncode == 2
            and json.loads(done.stdout)["error"]["code"] == "usage"
            and keyword in json.loads(done.stdout)["error"]["message"]
            and "--prompt-file" not in json.loads(done.stdout)["error"]["message"],
            f"{what} exits 2 usage with a message naming the input problem ({keyword!r}), not the prompt",
            "claude-session interface.md Exit codes (2: input-file names, --input-file -, files over input_max_bytes)",
            (done.returncode, done.stdout[-300:]),
        )
    check(
        len(world.json_tool("claude-session", "list")["sessions"]) == sessions,
        "no session was allocated by the refused starts",
        "claude-session interface.md Launch order (usage errors allocate nothing)",
    )
    # positive control: the same work dir and prompt with a small input starts, so the refusals
    # above come from the inputs and not from the setup
    small = fixtures / "small.txt"
    small.write_text("ok\n")
    control = cs_raw(world, *start_args(world, work, "agent=ag-input", "--input-file", str(small)))
    check(
        control.returncode == 0,
        "the same start with a small, well-named input succeeds (positive control)",
        "claude-session interface.md Launch flags (--input-file)",
        (control.returncode, control.stdout[-300:]),
    )
    world.note_session(json.loads(control.stdout)["csession_id"])


@SC("cs-wait-interval")
def test_wait_has_no_interval_flag_and_cron_install_interval_is_one_to_fifty_nine(
    world: World,
) -> None:
    """claude-session interface.md ``wait`` (``claude-session wait <id> [--run] [--timeout]
    [--outcome-file]``: no polling flag; the poll period is the config's ``poll_interval``) and
    ``cron install`` (``--interval`` 1 to 59 minutes, else exit 2; validated before any crontab is read, so the tripwire crontab is never reached).
    The #36 comment's "not covered: ``wait --interval``" names a flag claude-session does not have
    (a doc bug in docs/live-testing.md, corrected there): the test pins that the real tool refuses
    it, so a fake or a caller cannot rely on it, and checks the range of the ``--interval`` that does exist (``cron install``).
    agent-spawn's own ``wait`` has none either. #116."""
    work = prepare(world, "lost_grace: 600s\npoll_interval: 100ms\n")
    started = cs(world, *start_args(world, work, "agent=ag-wait"))
    csid = started["csession_id"]
    refused = cs_raw(world, "wait", csid, "--interval", "1", "--timeout", "1s")
    world.observe(
        "claude-session.wait.interval", (refused.returncode, refused.stdout, refused.stderr[-200:])
    )
    check(
        refused.returncode == 2,
        "claude-session wait --interval 1 is a usage error (exit 2): the flag does not exist",
        "claude-session interface.md wait (flags: --run, --timeout, --outcome-file)",
        (refused.returncode, refused.stdout[-300:], refused.stderr[-300:]),
    )
    ours = world.agent_spawn("wait", "ag-aaaaaaaa", "--interval", "1", "--json")
    check(
        ours.returncode == 2,
        "agent-spawn wait --interval is a usage error too",
        "interface.md wait (no polling flag)",
        (ours.returncode, ours.stdout[-300:], ours.stderr[-300:]),
    )
    # the wait that does exist runs on poll_interval: a pending run is canceled by stop, and wait
    # returns the run's outcome promptly rather than after a long poll
    cs(world, "stop", csid)
    begun = time.monotonic()
    done = cs_raw(world, "wait", csid, "--timeout", "30s")
    world.observe(
        "claude-session.wait.after-stop",
        (done.returncode, round(time.monotonic() - begun, 2), done.stdout[-400:]),
    )
    check(
        done.returncode == WAIT_AFTER_STOP_EXIT
        and time.monotonic() - begun < 20
        and json.loads(done.stdout)["outcome"] == "not-started"
        and json.loads(done.stdout)["run_exit_code"] == WAIT_AFTER_STOP_EXIT,
        "wait of a run that ended by stop returns at once with exit 12 and the run's outcome not-started",
        "claude-session interface.md wait (exit code of the run's outcome)",
        (done.returncode, done.stdout[-300:]),
    )
    for interval in ("0", "60", "abc"):
        bad = world.tool(
            "claude-session", "cron", "install", "--interval", interval, "--dry-run", "--json"
        )
        world.observe(
            f"claude-session.cron.interval.{interval}", (bad.returncode, bad.stdout[-200:])
        )
        check(
            bad.returncode == 2
            and "--interval" in bad.stdout
            and (interval == "abc" or "1 and 59" in bad.stdout),
            f"cron install --interval {interval} is a usage error whose message names --interval (and, for 0 and 60, the 1 to 59 range)",
            "claude-session interface.md cron install (--interval 1 to 59, else exit 2)",
            (bad.returncode, bad.stdout[-200:]),
        )
    # the bounds are accepted (against a fake crontab: a valid interval does read the table, the
    # tripwire stub would exit 70)
    write_fake_crontab(world.bin / "crontab", world.crontab_table)
    for interval in ("1", "59"):
        world.crontab_table.unlink(missing_ok=True)
        good = world.tool("claude-session", "cron", "install", "--interval", interval, "--json")
        table = world.crontab_table.read_text() if world.crontab_table.exists() else ""
        check(
            good.returncode == 0 and f"*/{interval} * * * * " in table,
            f"cron install --interval {interval} is accepted and the job runs every {interval} minute(s)",
            "claude-session interface.md cron install (--interval 1 to 59)",
            (good.returncode, good.stdout[-300:], table[-300:]),
        )


def run_on_tty(
    world: World,
    argv: Sequence[str],
    *,
    typed: str = "",
    timeout: float = 60,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` with a pseudo-terminal as its stdin (so ``isatty(0)`` is true), in the private
    environment, in its own session. ``typed`` is what a human types at the prompt: written to the
    terminal's master side up front (the line discipline holds it until the program reads). stdout
    and stderr stay pipes, so ``--json`` output is captured. On a timeout exactly the group this
    call created is killed."""
    env = world.env()
    master, slave = pty.openpty()
    try:
        try:
            child = subprocess.Popen(
                list(argv),
                env=env,
                cwd=cwd if cwd is not None else world.root / "repos",
                stdin=slave,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        finally:
            os.close(slave)
        try:
            if typed:
                os.write(master, typed.encode())
            out, err = child.communicate(timeout=world.timeout(timeout))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                for pipe in (child.stdout, child.stderr):
                    if pipe is not None:
                        pipe.close()
                child.wait(timeout=5)
            raise LiveError(
                f"timed out after {timeout:.0f}s on a tty: {' '.join(argv[:4])}"
            ) from None
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            raise
    finally:
        os.close(master)
    return subprocess.CompletedProcess(list(argv), child.returncode, out, err)


@SC("tty-human-gates")
def test_human_gates_and_yes_at_a_real_terminal(world: World) -> None:
    """wtctl interface.md Audit and log (Actor: ``--actor`` absent and stdin a TTY gives the human
    actor ``human:<me.name>``; human-only commands exit 13 without one), Remove order step 7 and
    ``baseline accept`` (from a TTY without ``--yes`` wtctl asks and a "no" changes nothing;
    ``--yes`` skips only the prompt), and agent-spawn interface.md Global flags (Human-only flags:
    accepted when stdin is a TTY, and the default actor is then ``human:<me.name>``; an explicit
    non-human ``--actor`` is refused even at a TTY). Every other live test stands in for the
    terminal with ``--actor human:tester``; here a pseudo-terminal is the stdin and no ``--actor``
    is given. #116 (a human ``--yes`` at a TTY)."""
    payload = world.root / "tty-payload.json"
    escalate = write_script(world, "tty-escalate.sh", f"cat > {payload}; exit 13\n")
    wt = stub_setup(world, "exit 0", agent_spawn_config=hook_config("pre-launch", (escalate, None)))
    made = wt.create("tty1", "ag-aaaa")["record"]
    rid = made["id"]
    wtctl = str(world.bin / "wtctl")
    wtctl_db_update(world, "update records set baseline_digest = ? where id = ?", "0" * 64, rid)

    # no terminal, no actor: a human-only command exits 13
    plain = wt.run("baseline", "accept", rid, "--yes")
    check(
        plain.returncode == 13 and err_of(plain)["code"] == "confirmation_required",
        "without a TTY and without an actor baseline accept --yes exits 13 confirmation_required",
        "wtctl interface.md Audit and log (human-only commands exit 13 without a human actor)",
        (plain.returncode, plain.stdout[-300:]),
    )
    # a terminal, no --yes: the prompt appears on stderr; "n" declines and changes nothing
    declined = run_on_tty(world, [wtctl, "--json", "baseline", "accept", rid], typed="n\n")
    world.observe(
        "wtctl.tty.declined", (declined.returncode, declined.stdout[-200:], declined.stderr[-300:])
    )
    check(
        declined.returncode == 13
        and "[y/N]" in declined.stderr
        and wt.run("baseline", "check", rid).returncode == 20,
        "at a TTY without --yes baseline accept asks [y/N]; 'n' exits 13 and leaves the baseline unchanged",
        "wtctl interface.md baseline accept (asks unless --yes), Remove order step 7",
        (declined.returncode, declined.stderr[-300:]),
    )
    # a terminal and --yes, still no --actor: the actor defaults to a human and the accept goes through
    # (wtctl 1.0.0 writes no audit record for it, see test_baseline_accept_is_audited)
    accepted = run_on_tty(world, [wtctl, "--json", "baseline", "accept", rid, "--yes"])
    check(
        accepted.returncode == 0
        and "[y/N]" not in accepted.stderr
        and wt.run("baseline", "check", rid).returncode == 0,
        "at a TTY with --yes and no --actor baseline accept runs as the default human actor, without a prompt",
        "wtctl interface.md Audit and log (Actor: the default is human:<me.name> when stdin is a TTY)",
        (accepted.returncode, accepted.stdout[-300:], accepted.stderr[-300:]),
    )

    # agent-spawn: --override-escalation is accepted at a TTY (default human actor) and refused with an auto actor
    stub = wt
    spawn = [str(world.agent_spawn_path), "--json"]
    refused = run_on_tty(
        world, [*spawn, *launch_args(stub, "--override-escalation", "--actor", "auto:bot")]
    )
    check(
        refused.returncode == 2 and json.loads(refused.stdout)["error"]["code"] == "usage",
        "an explicit non-human --actor is refused for --override-escalation even at a TTY",
        "interface.md Global flags (an explicit non-human --actor is refused even when stdin is a TTY)",
        (refused.returncode, refused.stdout[-300:]),
    )
    ran = run_on_tty(
        world, [*spawn, *launch_args(stub, "--override-escalation", "--wait")], timeout=120
    )
    world.observe("agent-spawn.tty.override", (ran.returncode, ran.stdout[-300:]))
    parked_id = json.loads(ran.stdout)["agent"]["agent_id"]
    body = json.loads(payload.read_text())
    check(
        ran.returncode == 14
        and body["override_escalation"] is True
        and str(body["actor"]).startswith("human:"),
        "at a TTY --override-escalation needs no --actor: the escalation counts as allow and the hook payload names a human actor",
        "interface.md Global flags (Human-only flags; default actor human:<me.name> when stdin is a TTY), Escalation overrides",
        (ran.returncode, {k: body.get(k) for k in ("override_escalation", "actor")}),
    )
    # the cli gate itself: teardown --force is human-only, an explicit auto actor is refused at a TTY
    gated = run_on_tty(world, [*spawn, "teardown", parked_id, "--force", "--actor", "auto:bot"])
    world.observe("agent-spawn.tty.teardown-force-auto", (gated.returncode, gated.stdout[-300:]))
    check(
        gated.returncode == 2
        and json.loads(gated.stdout)["error"]["code"] == "usage"
        and show(world, parked_id)["agent"]["status"] == "parked",
        "teardown --force with --actor auto:bot is refused at a TTY (exit 2 usage) and changes nothing",
        "interface.md Global flags (Human-only flags: an explicit non-human --actor is refused even at a TTY)",
        (gated.returncode, gated.stdout[-300:]),
    )


@SC("wtctl-baseline-no-classes")
@pytest.mark.xfail(
    strict=True,
    raises=KnownDivergence,
    reason="wtctl gap, wtctl#5 (closed; the TODO(#5) in baseline_accept.py remains; no open issue): "
    "wtctl 1.0.0 does not audit baseline accept although interface.md Audit and log lists the "
    "baseline_accept action",
)
def test_baseline_accept_is_audited(world: World) -> None:
    """wtctl interface.md Audit and log (``audit.jsonl`` actions: ``baseline_accept`` with the old
    and new digest, the changed items and the anchors, and the actor). Observed with wtctl 1.0.0:
    ``export --audit`` is empty after a human accept. strict: promote this to a plain test when
    wtctl audits the action."""
    wt = Wt(world)
    made = wt.create("au1", "ag-aaaa")["record"]
    wtctl_db_update(
        world, "update records set baseline_digest = ? where id = ?", "0" * 64, made["id"]
    )
    done = wt.run("baseline", "accept", made["id"], "--yes", "--actor", "human:tester")
    check(
        done.returncode == 0,
        "the human accept succeeds",
        "wtctl baseline accept",
        done.stdout[-300:],
    )
    records = json.loads(wt.run("export", "--audit").stdout)
    if "baseline_accept" not in [r["action"] for r in records]:
        raise KnownDivergence(
            f"the accept succeeded but the audit records lack baseline_accept (wtctl#5): {records}"
        )
    check(
        [r["action"] for r in records] == ["baseline_accept"]
        and records[0]["actor"] == "human:tester",
        "the accept is recorded in audit.jsonl with the human actor",
        "wtctl interface.md Audit and log (baseline_accept)",
        records,
    )


def _status_run(world: World, csid: str) -> dict[str, Any]:
    return dict(world.json_tool("claude-session", "status", csid)["run"])


@SC("as-window-close-post-park")
def test_a_late_reconcile_after_a_window_close_park_writes_no_second_post_park(
    world: World,
) -> None:
    """#75 item 4 (a late ``reconcile`` after a park delivers no second ``post-park``) on the two
    window park paths the billed ``interactive-window-close-unreported`` and
    ``window-wrapper-dead-child-alive-real`` reach with a real claude: an unreported session whose
    window is killed (parked ``killed``, ``forwarded_hup``), and SIGKILL of the recorded ``exec``
    under a child that dies with its terminal (the run is ``lost``, parked ``lost`` with attention
    ``lost``). Unbilled twin with a stub claude that dies on hangup: it checks the product's
    behavior on those paths; the two billed scenarios themselves still configure no ``post-park``
    hook (docs/live-testing.md, Not covered). interface.md Hooks (``post-park`` once per park),
    Reconcile."""
    hook_log = world.root / "post-park.log"
    wt = stub_setup(
        world,
        "while :; do sleep 0.2; done",
        agent_spawn_config=f'hooks:\n  post-park:\n    - cmd: "cat > /dev/null; echo ran >> {hook_log}"\n',
    )

    def runs() -> int:
        return hook_log.read_text().count("ran\n") if hook_log.exists() else 0

    # 1. the window of an unreported session is killed
    first = world.spawn_json(*window_launch_args(wt), timeout=120)["agent"]
    aid, csid = first["agent_id"], first["csession_id"]
    world.note_session(csid)
    state = wait_status(world, aid, "live")
    world.track(int(state["supervisor"]["pid"]), tree=True)
    run = wait_for(
        lambda: (r := _status_run(world, csid)).get("child_alive") and r,
        30,
        "the stub child to be running",
    )
    child = int(run["child_pid"])
    pane = panes_of(world, aid)[0]
    killed = world.tmux("kill-window", "-t", pane["window_id"])
    check(killed.returncode == 0, "kill-window works", "design.md B3", killed.stderr)
    parked = wait_status(world, aid, "parked", 90)["agent"]
    wait_for(lambda: pid_gone(child), 30, "the stub to end after the hangup")
    check(
        parked["park_reason"] == "killed" and parked["needs_attention"] is None,
        "a killed window parks the unreported agent killed without attention",
        "interface.md Outcome table (killed, forwarded_hup)",
        {k: parked[k] for k in ("park_reason", "needs_attention", "last_end_reason")},
    )
    wait_for(lambda: runs() >= 1, 15, "post-park to run for the window-close park")
    world.spawn_json("reconcile", ok=(0, 12))
    world.spawn_json("reconcile", ok=(0, 12))
    time.sleep(3)  # bounded settle: a second delivery would be asynchronous
    check(
        runs() == 1,
        "post-park ran exactly once for the window-close park; late reconciles delivered no second one",
        "interface.md Hooks (post-park once per park), #75 item 4",
        runs(),
    )
    world.spawn_json("teardown", aid)

    # 2. SIGKILL of the recorded exec under a child that dies with its terminal
    second = world.spawn_json(*window_launch_args(wt), timeout=120)["agent"]
    bid, bcsid = second["agent_id"], second["csession_id"]
    world.note_session(bcsid)
    state = wait_status(world, bid, "live")
    world.track(int(state["supervisor"]["pid"]), tree=True)
    run = wait_for(
        lambda: (
            (r := _status_run(world, bcsid)).get("child_alive") and r.get("wrapper_alive") and r
        ),
        30,
        "the stub child and its wrapper to be running",
    )
    wrapper, child = int(run["wrapper_pid"]), int(run["child_pid"])
    check(
        state["supervisor"].get("exec_pid") == wrapper,
        "the exec pid agent-spawn recorded is claude-session's wrapper pid",
        "design.md D19 / claude-session status run.wrapper_pid",
        (state["supervisor"], wrapper),
    )
    before = runs()
    signal_exact(world, wrapper, signal.SIGKILL, comm_contains="exec")
    wait_for(lambda: pid_gone(wrapper), 20, "the wrapper to be dead")
    wait_for(lambda: pid_gone(child), 30, "the stub to end with its terminal")
    world.spawn_json("reconcile", ok=(0, 12))
    parked = wait_status(world, bid, "parked", 90)["agent"]
    world.observe(
        "window-wrapper-dead.stub-dies",
        {
            k: parked[k]
            for k in ("park_reason", "needs_attention", "supervisor_lost", "last_outcome")
        },
    )
    check(
        parked["park_reason"] == "lost" and parked["needs_attention"] == "lost",
        "a child that died with its terminal after the wrapper was killed ends the run lost and parks the agent lost with attention lost",
        "interface.md Outcome table (lost), Liveness",
        {k: parked[k] for k in ("park_reason", "needs_attention", "supervisor_lost")},
    )
    wait_for(lambda: runs() >= before + 1, 15, "post-park to run for the lost park")
    world.spawn_json("reconcile", ok=(0, 12))
    time.sleep(3)  # bounded settle: a second delivery would be asynchronous
    check(
        runs() == before + 1,
        "post-park ran once for the lost park; a late reconcile delivered no second one",
        "interface.md Hooks (post-park once per park), #75 item 4",
        (before, runs()),
    )
    world.spawn_json("teardown", bid, ok=(0, 20))


@SC("wtctl-version")
def test_wtctl_prints_its_release_version(world: World) -> None:
    """wtctl interface.md Global flags (``--version`` prints ``wtctl <version>``). The 0.32.x
    builds printed the hard-coded ``0.0.0`` placeholder, so the harness recorded the version as
    unknown; a released 1.0.0 prints its real version (#116). This test is for released installs: it
    skips (never fails, as docs/live-testing.md says of unknown versions) on a build that prints
    ``0.0.0`` or is below the minimum, fails when a version of 1.0.0 or later prints another line,
    and the harness's own reading of the version (``tools-versions``) must agree with it."""
    done = world.tool("wtctl", "--version")
    world.observe("wtctl.version", done.stdout.strip())
    have = parse_version(done.stdout)
    minimum = parse_version(MINIMUMS["wtctl"])
    assert minimum is not None
    check(
        have is not None,
        "wtctl --version prints a version that parses",
        "wtctl interface.md Global flags (--version)",
        (done.returncode, done.stdout, done.stderr),
    )
    assert have is not None
    if have == (0, 0, 0) or have < minimum:
        pytest.skip(
            f"wtctl prints {done.stdout.strip()!r}: a pre-release version "
            "(0.0.0 or below the minimum); this test is for released installs"
        )
    found = re.fullmatch(r"wtctl (\d+)\.(\d+)\.(\d+)\S*", done.stdout.strip())
    if have < (1, 0, 0) and (done.returncode != 0 or found is None):
        pytest.skip(f"wtctl {have} predates the 1.0.0 version line: {done.stdout.strip()!r}")
    check(
        done.returncode == 0 and found is not None,
        "wtctl --version prints 'wtctl <major>.<minor>.<patch>'",
        "wtctl interface.md Global flags (--version)",
        (done.returncode, done.stdout, done.stderr),
    )
    assert found is not None
    parsed = tuple(int(g) for g in found.groups())
    tool = world.tools["wtctl"]
    check(
        tool is not None and tool.version == parsed,
        "the harness reads the same version as wtctl --version (it is not recorded as unknown)",
        "tests/live/harness.py read_version",
        (tool.version if tool else None, parsed),
    )
