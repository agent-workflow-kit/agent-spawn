"""claude-session behaviors agent-spawn relies on, against the real claude-session (no Claude Code).

A run that nothing adopted ("pending") needs no Claude process at all; where a program called
``claude`` must run (``exec``), a stub script stands in (``World.install_claude(stub_body=...)``):
the stub is not Claude Code, so the tests check claude-session's own bookkeeping only. The
claude-session config is private (``lost_grace`` and ``kill_grace`` shortened per test).
"""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from .harness import World, check, wait_for

SC = pytest.mark.live_scenario


def cs(world: World, *args: str, ok: tuple[int, ...] = (0,), **kw: Any) -> Any:
    return world.json_tool("claude-session", *args, ok=ok, **kw)


def cs_raw(world: World, *args: str, **kw: Any) -> subprocess.CompletedProcess[str]:
    """claude-session takes ``--json`` after the command (a leading ``--json`` is exit 2)."""
    rest = [a for a in args if a != "--json"]
    return world.tool("claude-session", *rest, "--json", **kw)


def prepare(world: World, config: str = "lost_grace: 600s\n") -> Path:
    """A private claude-session config, a stub claude and a prompt file; returns the cwd."""
    world.install_claude(stub_body="exit 3")
    world.claude_session_config(config)
    work = world.root / "work"
    work.mkdir(exist_ok=True)
    (world.root / "prompt.md").write_text("reply with the single word ok\n")
    return work


def start_args(world: World, work: Path, label: str, *extra: str) -> list[str]:
    return [
        "start",
        "--kind",
        "implement",
        "--mode",
        "headless",
        "--prompt-file",
        str(world.root / "prompt.md"),
        "--label",
        label,
        "--cwd",
        str(work),
        "--model",
        "haiku",
        *extra,
    ]


def start(world: World, work: Path, label: str, *extra: str) -> dict[str, Any]:
    return cs(world, *start_args(world, work, label, *extra))


@SC("cs-pending-run")
def test_a_pending_run_status_list_stop_and_forget(world: World) -> None:
    """#36 #9 (status right after start: ``current_run == run``, pending, ``run.cwd`` null;
    ``list --label`` ANDs repeated labels and excludes forgotten sessions; ``stop`` of a pending run
    exits 0 ``canceled: true`` and writes the outcome file once, mode 0600; forget of a pending
    session exits 20 ``live`` and afterwards the id is unknown, 10), #36 #14 (``list`` and ``status``
    carry the labels agent-spawn compares), #36 #15 (status fields)."""
    work = prepare(world)
    outcome = world.root / "outcome" / "o.json"
    outcome.parent.mkdir()
    started = start(world, work, "agent=ag-pend1", "--label", "job=j1", "--outcome-file", str(outcome))
    sid = started["csession_id"]
    check(
        started["argv"][1:3] == ["exec", "--session-id"] and started["run"] == 1,
        "start prints the exec argv that adopts the pending run",
        "#36 #27 / design.md uses (exec --session-id)",
        started["argv"],
    )
    check(Path(started["csession_dir"]).is_dir(), "start creates the session directory", "#36 #27")
    status = cs(world, "status", sid)
    run = status["run"]
    check(
        status["current_run"] == started["run"] and status["pending"] is True,
        "right after start current_run == run and the session is pending",
        "#36 #9 status",
        (status["current_run"], status["pending"]),
    )
    check(
        run["phase"] == "launch" and run["state"] == "starting" and run["cwd"] is None,
        "a pending run has phase launch, state starting and a null run.cwd",
        "#36 #9 status / design.md D43",
        {k: run.get(k) for k in ("phase", "state", "cwd")},
    )
    check(status["expires_at"], "a pending run has expires_at", "#36 #29 (B) real status of an unadopted run")
    check(
        {"agent=ag-pend1", "job=j1", "kind=implement"} <= set(status["labels"]),
        "status carries the labels agent-spawn compares",
        "#36 #14/#15 (labels in status)",
        status["labels"],
    )
    for key in ("tainted", "csession_dir", "claude_session_id", "live"):
        check(key in status, f"status carries {key}", "#36 #15 (self-labeling reads status)", sorted(status))
    check(
        run.get("reads_third_party") is False and "implicit_done" in run,
        "status carries reads_third_party and implicit_done",
        "#36 #15 / #88",
        sorted(run),
    )
    world.observe("claude-session.status.keys", sorted(status))
    world.observe("claude-session.status.run.keys", sorted(run))
    other = start(world, work, "agent=ag-pend2", "--label", "job=j1")["csession_id"]
    both = cs(world, "list", "--label", "agent=ag-pend1", "--label", "job=j1")["sessions"]
    check([s["csession_id"] for s in both] == [sid], "repeated --label filters AND", "#36 #9 list --label")
    check(
        len(cs(world, "list", "--label", "job=j1")["sessions"]) == 2,
        "list --label job=j1 lists both",
        "#36 #9 list",
    )
    check(
        all(s["pending"] is True for s in cs(world, "list")["sessions"]),
        "list reports pending: true for pending runs",
        "FIDELITY.md (list and status agree about a pending run)",
    )
    short = cs_raw(world, "--json", "status", "cs-")
    check(
        short.returncode == 2,
        "an id prefix needs at least 4 characters after cs- (exit 2); an ambiguous prefix (21) "
        "needs two ids sharing four characters and is not provoked here",
        "#36 #8 claude-session (ambiguous prefix exit 21: deferred, see docs/live-testing.md)",
        (short.returncode, short.stdout[-200:]),
    )
    unknown_prefix = cs_raw(world, "--json", "status", "cs-zzzz")
    check(unknown_prefix.returncode == 10, "an unknown prefix exits 10", "#36 #8 claude-session")
    # forget of a pending session is refused
    live = cs_raw(world, "--json", "forget", sid)
    err = json.loads(live.stdout)
    world.observe("claude-session.forget.pending", (live.returncode, live.stdout[-300:]))
    check(
        live.returncode == 20 or err.get("skipped"),
        "forget of a pending session exits 20 live",
        "#36 #9 forget (exit 20 live for pending or live sessions)",
        live.stdout[-300:],
    )
    # stop of a pending run cancels it and writes the outcome file once
    stopped = cs(world, "stop", sid)
    check(
        stopped["canceled"] is True and stopped["ended"] is True,
        "stop of a pending run exits 0 with canceled: true",
        "#36 #9 stop",
        stopped,
    )
    check(outcome.exists(), "stop wrote the outcome file", "#36 #9 stop (runs the relay and writes it)")
    mode = stat.S_IMODE(outcome.stat().st_mode)
    check(mode == 0o600, "the outcome file has mode 0600", "#36 #8 claude-session (outcome file mode 0600)", oct(mode))
    first = outcome.read_bytes()
    mtime = outcome.stat().st_mtime_ns
    cs_raw(world, "--json", "reconcile")
    cs_raw(world, "--json", "status", sid)
    check(
        outcome.read_bytes() == first and outcome.stat().st_mtime_ns == mtime,
        "the outcome file is written once",
        "#36 #8 claude-session (every finalizer writes the outcome file once)",
    )
    body = json.loads(first)
    world.observe("claude-session.outcome-file.keys", sorted(body))
    check(body.get("outcome") in ("killed", "not-started", "canceled"), "outcome object", "#36 #75 2", body)
    done = cs(world, "status", sid)
    world.observe("claude-session.stop-pending.final", {k: done["run"].get(k) for k in ("outcome", "end_reason", "phase")})
    # forget, then the id is unknown for every command
    cs(world, "forget", sid)
    for argv in (["status", sid], ["label", "add", sid, "x=y"], ["label", "remove", sid, "x=y"], ["stop", sid], ["forget", sid]):
        gone = cs_raw(world, "--json", *argv)
        check(
            gone.returncode == 10,
            f"after forget `{argv[0]}` treats the id as unknown (10)",
            "#36 #9 forget / #36 #81 label of a forgotten session exits 10",
            (argv, gone.returncode, gone.stdout[-150:]),
        )
    check(
        [s["csession_id"] for s in cs(world, "list", "--label", "agent=ag-pend1")["sessions"]] == [],
        "list excludes forgotten sessions",
        "#36 #9 list",
    )
    del other


@SC("cs-pending-run")
def test_an_unadopted_run_expires_never_adopted(world: World) -> None:
    """#36 #75 1 and promotion check B: after ``lost_grace`` an unadopted pending run is
    ``not-started`` / ``never_adopted`` in phase ``launch`` (status finalizes it), the pending flag
    and ``expires_at`` clear; a resume dry run of it and a fork dry run show the documented shapes."""
    work = prepare(world, "lost_grace: 3s\n")
    started = start(world, work, "agent=ag-exp1")
    sid = started["csession_id"]
    time.sleep(4.5)
    status = cs(world, "status", sid)
    run = status["run"]
    check(
        run["outcome"] == "not-started" and run["end_reason"] == "never_adopted" and run["phase"] == "launch",
        "an unadopted run ends not-started / never_adopted in phase launch after lost_grace",
        "#36 #75 1 / promotion check B (expired, never_adopted)",
        {k: run.get(k) for k in ("outcome", "end_reason", "phase")},
    )
    check(status["pending"] is False and status["expires_at"] is None, "pending and expires_at clear", "#36 #29 (B)")
    listed = cs(world, "list", "--label", "agent=ag-exp1")["sessions"]
    check(
        listed[0]["last_outcome"] == "not-started" and listed[0]["last_end_reason"] == "never_adopted",
        "list reports the finalized run",
        "#36 #9 list may finalize runs",
        listed[0],
    )
    plan = cs(world, "start", "--resume", sid, "--dry-run", "--cwd", str(work), "--mode", "headless", ok=(0, 20))
    world.observe("claude-session.resume-dry-run.not-started", {k: plan.get(k) for k in ("csession_id", "cwd", "taints", "would_refuse", "fork")})
    check(plan.get("csession_id") == sid, "a resume dry run names the existing session", "#36 #9 dry run")
    fork = cs_raw(world, "--json", "start", "--resume", sid, "--fork", "--dry-run", "--cwd", str(work), "--mode", "headless")
    world.observe("claude-session.fork-dry-run.not-started", (fork.returncode, fork.stdout[-400:]))
    if fork.returncode == 0:
        check(
            json.loads(fork.stdout)["csession_id"] is None and json.loads(fork.stdout)["fork"] is True,
            "a fork dry run has csession_id null",
            "#36 #9 dry run (fork dry run has csession_id: null)",
        )
    cs(world, "forget", sid)
    check(
        not cs(world, "list", "--label", "agent=ag-exp1")["sessions"],
        "a forgotten, finalized session leaves list",
        "#36 #9",
    )


@SC("cs-pending-run")
def test_verify_secret_reconcile_and_dry_run_refusal(world: World) -> None:
    """#36 #9 (``verify-secret`` returns 11 for a pending run, strips trailing whitespace, needs
    stdin; ``reconcile --dry-run`` exits 0 with the ``reconcile.json`` shape, takes no lease;
    a resume dry run of a live session exits 20 with ``would_refuse: live``), #36 #15
    (verify-secret stdin protocol and exit codes), #36 #81."""
    work = prepare(world)
    sid = start(world, work, "agent=ag-vs1")["csession_id"]
    for secret in ("wrong-secret\n", "wrong-secret"):
        done = cs_raw(world, "--json", "verify-secret", sid, input_text=secret)
        world.observe("claude-session.verify-secret.pending", (done.returncode, done.stdout[-200:]))
        check(
            done.returncode == 11,
            "verify-secret of a pending run (no secret yet) returns 11",
            "#36 #9 verify-secret (11 for a pending run) / #36 #15",
            (done.returncode, done.stdout[-200:]),
        )
    unknown = cs_raw(world, "--json", "verify-secret", "cs-nonexist1", input_text="x\n")
    world.observe("claude-session.verify-secret.unknown", (unknown.returncode, unknown.stdout[-100:]))
    check(
        unknown.returncode == 11 and json.loads(unknown.stdout) == {"match": False},
        "verify-secret of an unknown session answers like a wrong secret (11, match false): "
        "it is not an existence oracle",
        "#36 #15 verify-secret exit codes (the fake/interface assumed 10)",
        (unknown.returncode, unknown.stdout[-100:]),
    )
    plan = cs_raw(world, "--json", "start", "--resume", sid, "--dry-run", "--cwd", str(work), "--mode", "headless")
    body = json.loads(plan.stdout)
    check(
        plan.returncode == 20 and body.get("would_refuse") == "live",
        "a resume dry run of a live (pending) session exits 20 with would_refuse live",
        "#36 #9 dry run",
        (plan.returncode, plan.stdout[-300:]),
    )
    check(
        isinstance(body.get("profile", {}).get("reads_third_party"), bool) and isinstance(body.get("taints"), bool),
        "refusal plans carry boolean profile.reads_third_party and taints",
        "#36 #9 (exit 12 plans still carry them)",
        {k: body.get(k) for k in ("taints",)},
    )
    reconcile = cs(world, "reconcile", "--dry-run")
    for key in ("already_running", "dry_run", "changes", "orphans", "warnings"):
        check(key in reconcile, f"reconcile --dry-run carries {key}", "#36 #81 (reconcile.json shape)", sorted(reconcile))
    check(reconcile["dry_run"] is True and reconcile["already_running"] is False, "dry run takes no lease", "#36 #81")
    cs(world, "stop", sid)
    cs(world, "forget", sid)


@SC("cs-start-killed")
def test_a_killed_start_never_leaves_an_unlabeled_session(world: World) -> None:
    """#36 #75 2: SIGKILL ``claude-session start`` at many offsets (before allocation, between
    allocation and freezing, after freezing); ``list`` must never show a session without the
    ``agent=`` label the start was given, and a label search finds none or exactly one."""
    work = prepare(world)
    argv_base = [str(world.bin / "claude-session"), *start_args(world, work, "agent=PLACEHOLDER", "--json")]
    seen = 0
    for i, offset in enumerate(range(0, 2000, 100)):
        label = f"agent=ag-kill{i:02d}"
        argv = [a if a != "agent=PLACEHOLDER" else label for a in argv_base]
        child = subprocess.Popen(
            argv,
            env=world.env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(offset / 1000)
        try:
            os.kill(child.pid, signal.SIGKILL)  # our own, unreaped child: the PID cannot be reused
        except ProcessLookupError:
            pass
        child.wait()
        found = cs(world, "list", "--label", label)["sessions"]
        check(len(found) <= 1, "a label search finds none or exactly one session", "#36 #75 2", (label, len(found)))
        seen += len(found)
    everything = cs(world, "list")["sessions"]
    unlabeled = [s["csession_id"] for s in everything if not any(label.startswith("agent=ag-kill") for label in s["labels"])]
    check(
        not unlabeled,
        "a killed start never leaves a session without the label it was given",
        "#36 #75 2 (never an unlabeled one)",
        unlabeled,
    )
    world.observe("claude-session.killed-start.sessions-created", f"{seen} of 20")
    for s in everything:
        status = cs(world, "status", s["csession_id"])
        if status["run"]["outcome"] not in (None,):
            world.observe(f"claude-session.killed-start.{s['csession_id']}", {k: status["run"].get(k) for k in ("outcome", "end_reason", "phase")})


@SC("cs-dry-run-shapes")
def test_implicit_done_flags_in_the_dry_run_plan(world: World) -> None:
    """#36 #88 (needs claude-session >= 0.43.0): both flags accepted by ``start``; the dry-run
    ``implicit_done`` shows the effective value; both together, ``=value`` and ``--reset
    implicit-done`` with a flag: the exit codes are recorded."""
    work = prepare(world)
    base = ["start", "--dry-run", "--kind", "implement", "--mode", "headless", "--prompt-file", str(world.root / "prompt.md"), "--cwd", str(work)]
    plain = cs(world, *base)
    yes = cs(world, *base, "--implicit-done")
    no = cs(world, *base, "--no-implicit-done")
    check(
        yes["implicit_done"] is True and no["implicit_done"] is False,
        "the dry run's implicit_done is the effective value of the flag",
        "#36 #88 (real dry-run implicit_done)",
        (yes["implicit_done"], no["implicit_done"]),
    )
    check(plain["implicit_done"] is plain["profile"]["implicit_done"], "without a flag the kind's value applies", "#36 #88")
    origin = plain.get("origins", {})
    world.observe("claude-session.dry-run.origins.implicit_done", {k: origin.get(k) for k in ("implicit_done",)})
    for label, extra in (
        ("both flags", ["--implicit-done", "--no-implicit-done"]),
        ("=value", ["--implicit-done=true"]),
        ("reset with flag", ["--reset", "implicit-done", "--no-implicit-done"]),
    ):
        done = cs_raw(world, "--json", *base, *extra)
        world.observe(f"claude-session.implicit-done.{label}", (done.returncode, done.stdout[-160:]))
    value = cs_raw(world, "--json", *base, "--implicit-done=true")
    check(value.returncode == 2, "--implicit-done=value exits 2", "#36 #88 / FIDELITY.md BooleanOptionalAction", value.returncode)
    both = cs_raw(world, "--json", *base, "--implicit-done", "--no-implicit-done")
    check(both.returncode == 0, "both flags together are accepted (the last wins)", "FIDELITY.md implicit-done", both.returncode)
    check(json.loads(both.stdout)["implicit_done"] is False, "the last of two flags wins", "FIDELITY.md implicit-done")


@SC("cs-dry-run-shapes")
def test_override_escalation_refusals(world: World) -> None:
    """#36 #78: ``start --resume --override-escalation`` exits 2 when the previous run did not end
    ``escalated``, with ``--fork``, and with a blank ``human:`` actor."""
    work = prepare(world, "lost_grace: 2s\n")
    sid = start(world, work, "agent=ag-esc1")["csession_id"]
    time.sleep(3.5)
    cs(world, "status", sid)
    base = ["start", "--resume", sid, "--dry-run", "--override-escalation", "--mode", "headless", "--cwd", str(work)]
    for label, extra in (("plain", []), ("fork", ["--fork"]), ("blank human actor", ["--actor", "human:"])):
        done = cs_raw(world, "--json", *base, *extra, extra={})
        world.observe(f"claude-session.override-escalation.{label}", (done.returncode, done.stdout[-200:]))
        check(
            done.returncode == 2,
            f"--override-escalation ({label}) exits 2 when the previous run did not end escalated",
            "#36 #78 (claude-session start --resume --override-escalation)",
            (done.returncode, done.stdout[-200:]),
        )


@SC("cs-stop-grace")
def test_stop_default_grace_matches_agent_spawns_default(world: World) -> None:
    """#36 #19 (``DEFAULT_GRACE``): claude-session's default ``kill_grace`` is the time ``stop``
    gives a SIGTERM-ignoring child before SIGKILL; agent-spawn's hard-coded wait budget must match.
    Also: an explicit ``--grace`` is honored, and a child that exits on SIGTERM is stopped at once
    (FIDELITY.md exec signals: SIGTERM then SIGKILL after the grace, exit 15)."""
    from agent_spawn import stop as spawn_stop

    work = prepare(world, "lost_grace: 600s\n")
    world.install_claude(stub_body="trap '' TERM; while :; do sleep 0.2; done")

    def run_and_stop(*stop_args: str) -> tuple[float, dict[str, Any], int]:
        started = start(world, work, "agent=ag-grace")
        child = subprocess.Popen(
            started["argv"],
            env=world.env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            wait_for(
                lambda: cs(world, "status", started["csession_id"])["run"].get("child_alive"),
                30,
                "the stub child of exec",
            )
            world.track(child.pid, tree=True)
            began = time.monotonic()
            done = cs(world, "stop", started["csession_id"], *stop_args)
            took = time.monotonic() - began
            child.communicate(timeout=30)
        except BaseException:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
        cs(world, "forget", started["csession_id"])
        return took, done, child.returncode

    took, done, code = run_and_stop()
    world.observe("claude-session.stop.default-grace", {"took": round(took, 2), "signals": done["signals"], "exit": code})
    check(
        [s["signal"] for s in done["signals"]] == ["SIGTERM", "SIGKILL"] and code == 15,
        "stop sends SIGTERM to the wrapper then SIGKILL to the child; exec exits 15",
        "FIDELITY.md exec signals (SIGTERM then SIGKILL after grace, exit 15)",
        done["signals"],
    )
    check(
        abs(took - spawn_stop.DEFAULT_GRACE) <= 3.0,
        "claude-session's default kill_grace equals agent-spawn's DEFAULT_GRACE "
        f"({spawn_stop.DEFAULT_GRACE:.0f} s), so stop does not give up before the escalation ends",
        "src/agent_spawn/stop.py DEFAULT_GRACE todo(#36)",
        f"stop took {took:.1f} s",
    )
    took2, _, _ = run_and_stop("--grace", "2s")
    check(1.5 <= took2 <= 6.0, "an explicit --grace 2s is honored", "design.md D37 / #36 #9 stop", took2)
    world.install_claude(stub_body="exit 0")
    started = start(world, work, "agent=ag-fast")
    quick = subprocess.run(
        started["argv"], env=world.env(), capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False
    )
    world.observe("claude-session.exec.stub-exit-0", (quick.returncode, quick.stdout[-300:], quick.stderr[-200:]))
    final = cs(world, "status", started["csession_id"])["run"]
    world.observe("claude-session.exec.stub-exit-0.final", {k: final.get(k) for k in ("outcome", "end_reason", "phase", "exit_code")})
    check(final["outcome"] is not None, "exec finalizes the run when the child exits", "FIDELITY.md exec")
