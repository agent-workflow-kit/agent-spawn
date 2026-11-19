"""termctl and tmux behaviors agent-spawn relies on, against the real programs (no Claude Code).

Every test runs in a ``World``: a private ``TMUX_TMPDIR`` (so termctl's default server is a
private one), a private HOME and state. Each check names the assumption it confirms and the
document that states it (``check(..., where)``); a docstring lists the FIDELITY.md / issue
comment items the run confirms so the later run can update those documents.
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from .harness import World, check, wait_for

SC = pytest.mark.live_scenario


def pane_pid_of(world: World, pane_id: str) -> int:
    done = world.tmux("display-message", "-p", "-t", pane_id, "#{pane_pid}")
    return int(done.stdout.strip())


def open_pane(world: World, *args: str, ok: tuple[int, ...] = (0,)) -> dict[str, Any]:
    return world.json_tool("termctl", "open", *args, ok=ok)


def panes(world: World, *args: str) -> list[dict[str, Any]]:
    return world.json_tool("termctl", "list", *args)["panes"]


@SC("tmux-real-server")
def test_tmux_commands_and_socket_options_agent_spawn_depends_on(world: World) -> None:
    """tmux items of the #8 comment (``list-commands``, ``-L`` with ``-S``, unknown-command text
    before a connection) and tmux >= 3.2 (``--keep-on-exit`` needs per-pane ``remain-on-exit``)."""
    listed = world.tmux("list-commands", socket=world.root / "a.sock")
    check(
        listed.returncode == 0 and "new-window" in listed.stdout,
        "tmux list-commands works without a server and lists new-window",
        "FIDELITY.md tmux (list-commands is fake_unsupported)",
        listed.stderr,
    )
    world.observe("tmux.list-commands.lines", len(listed.stdout.splitlines()))
    both = world.tmux("-L", "ignored-name", "-S", str(world.root / "b.sock"), "list-sessions")
    world.observe("tmux.-L with -S", (both.returncode, both.stderr.strip()))
    check(
        both.returncode == 1 and str(world.root / "b.sock") in both.stderr,
        "with -L and -S together the -S socket path is the one used",
        "#36 #8 tmux: -L with -S",
        both.stderr,
    )
    check(
        not (world.tmuxdir / f"tmux-{os.getuid()}" / "ignored-name").exists(),
        "-L with -S creates nothing under TMUX_TMPDIR",
        "#36 #8 tmux",
    )
    unknown = world.tmux("no-such-command", socket=world.root / "c.sock")
    world.observe("tmux.unknown-command", (unknown.returncode, unknown.stderr.strip()))
    check(
        unknown.returncode == 1 and "error connecting to" in unknown.stderr,
        "an unknown command without a server fails at the connect ('error connecting to'), "
        "not as 'unknown command' (the client only parses server-side commands there)",
        "FIDELITY.md tmux unknown command text",
        unknown.stderr,
    )
    semicolons = world.tmux(
        "start-server", ";", "display-message", "-p", "a", socket=world.root / "d.sock"
    )
    world.observe("tmux.start-server;display", (semicolons.returncode, semicolons.stdout))


@SC("termctl-open-pid")
def test_open_reports_the_pane_process_pid(world: World) -> None:
    """Spike A8 and #36 #26/#27: ``termctl open --json`` reports ``pid`` equal to tmux's
    ``#{pane_pid}``, which is the window process itself when the command execs in place (the
    peer check of the handoff), and the pane's PATH is what ``--env`` gives."""
    opened = open_pane(world, "--name", "a8", "--label", "agent=ag-live1", "--", "sleep", "300")
    pid = pane_pid_of(world, opened["pane_id"])
    check(
        opened["pid"] == pid,
        "open's pid equals #{pane_pid} (spike A8)",
        "design.md Live verification A8",
        (opened["pid"], pid),
    )
    check(
        opened["state"] == "running" and opened["socket_path"],
        "open reports running + socket",
        "#36 #27",
    )
    check(
        Path(opened["socket_path"]) == world.default_socket,
        "the default server is <TMUX_TMPDIR>/tmux-<uid>/default",
        "#36 #79",
        opened["socket_path"],
    )
    comm = Path(f"/proc/{pid}/comm").read_text().strip()
    check(
        comm == "sleep",
        "the pane process is the command itself, no wrapper shell",
        "design.md D21",
        comm,
    )

    # a wrapper shell that execs the real command keeps the same pid (supervise exec case)
    wrapped = open_pane(world, "--name", "wrap", "--", "sh", "-c", "exec sleep 300")
    wait_for(
        lambda: Path(f"/proc/{wrapped['pid']}/comm").read_text().strip() == "sleep",
        10,
        "the wrapper shell to exec sleep",
    )
    check(
        wrapped["pid"] == pane_pid_of(world, wrapped["pane_id"]),
        "a shell that execs keeps the pane pid (peer PID still matches after the exec)",
        "#36 #26: pid is a wrapper shell that execs supervise",
    )
    shell = open_pane(world, "--name", "noexec", "--", "sh", "-c", "sleep 300; true")
    world.track(shell["pid"], tree=True)
    child = Path(f"/proc/{shell['pid']}/comm").read_text().strip()
    check(
        child == "sh",
        "a pane command that does not exec stays a shell: pid is the shell, not its child",
        "#36 #29 (A): pid under default-shell wrapping",
        child,
    )
    # pid is the environment's: --env reaches the pane process
    envd = open_pane(world, "--name", "envd", "--env", "LIVE_X=1", "--", "sleep", "300")
    environ = Path(f"/proc/{envd['pid']}/environ").read_bytes().split(b"\0")
    check(
        b"LIVE_X=1" in environ,
        "--env reaches the pane process",
        "#8 termctl: pane process environment",
    )
    world.observe(
        "termctl.pane.environ.names", sorted(e.split(b"=")[0].decode() for e in environ if e)
    )


@SC("termctl-open-pid")
def test_dry_run_plan_against_the_real_open(world: World) -> None:
    """Promotion check E and #8 termctl: the ``--dry-run`` plan has the keys agent-spawn's
    preflight reads, creates nothing and names the same placement, session, labels and socket as
    the real open."""
    args = ["--name", "dry", "--label", "agent=ag-live2", "--keep-on-exit", "failed", "--"]
    plan = world.json_tool("termctl", "open", "--dry-run", *args, "sleep", "300")
    for key in (
        "dry_run",
        "placement",
        "session",
        "target",
        "socket_path",
        "labels",
        "argv",
        "tmux_commands",
    ):
        check(
            key in plan,
            f"the dry-run plan carries {key}",
            "#36 #29 (E) preflight drift",
            sorted(plan),
        )
    check(plan["dry_run"] is True and plan["argv"] == ["sleep", "300"], "dry-run argv", "#36 #8")
    check(not panes(world), "a dry run creates no pane", "design.md D36")
    real = open_pane(world, *args, "sleep", "300")
    for key in ("placement", "session", "target", "socket_path", "labels"):
        if key in real:
            check(
                plan[key] == real[key],
                f"the dry run and the real open agree on {key}",
                "#36 #29 (E): termctl open --dry-run vs the real open",
                (plan[key], real[key]),
            )
    world.observe("termctl.dry-run.keys", sorted(plan))
    world.observe("termctl.open.keys", sorted(real))


@SC("termctl-open-pid")
def test_bare_open_label_cap_and_argv_cap(world: World) -> None:
    """FIDELITY.md pinned divergence (bare ``open`` without ``--`` opens a shell pane, exit 0),
    #36 #27 label cap of 32 per pane and the 12288-byte ``open`` argv cap."""
    bare = world.tool("termctl", "--json", "open", "--name", "bare")
    check(
        bare.returncode == 0,
        "real termctl 0.25.0 opens a shell pane for open without -- (interface says exit 2; "
        "the fake follows the interface; FIDELITY.md DIVERGENCES)",
        "tests/fakebin/FIDELITY.md termctl",
        (bare.returncode, bare.stdout[-200:], bare.stderr[-200:]),
    )
    labels: list[str] = []
    for i in range(32):
        labels += ["--label", f"k{i}=v"]
    done = world.tool("termctl", "--json", "open", "--name", "l32", *labels, "--", "sleep", "300")
    check(done.returncode == 0, "32 labels are accepted", "#36 #27 label cap", done.stdout[-200:])
    over = world.tool(
        "termctl",
        "--json",
        "open",
        "--name",
        "l33",
        *labels,
        "--label",
        "x=y",
        "--",
        "sleep",
        "300",
    )
    check(
        over.returncode == 2,
        "33 labels exit 2",
        "#36 #27 label cap 32",
        (over.returncode, over.stdout[-200:]),
    )
    big = world.tool(
        "termctl", "--json", "open", "--name", "big", "--", "sleep", "300", "x" * 13000
    )
    world.observe("termctl.open.big-argv", (big.returncode, big.stdout[-150:]))
    check(big.returncode == 2, "an open argv over 12288 bytes exits 2", "#36 #27", big.returncode)


@SC("termctl-sighup-leader")
def test_closing_a_pane_hangs_up_only_the_pane_process(world: World, tmp_path: Path) -> None:
    """Spike B1/B2 regression: ``kill-pane`` (and ``termctl close --force``) deliver SIGHUP to the
    pane's process only; its child gets nothing while that process lives. This is why
    ``supervise`` forwards SIGHUP (D17)."""
    script = world.root / "b1.sh"
    script.write_text(
        'D="$1"\n'
        '( trap \'echo hup > "$D/child-hup"\' HUP; echo $$ > "$D/child.pid"; '
        "while :; do sleep 0.2; done ) &\n"
        "trap 'echo hup > \"$D/leader-hup\"; sleep 3; exit 0' HUP\n"
        'echo $$ > "$D/leader.pid"\n'
        "while :; do sleep 0.2; done\n"
    )
    out = world.root / "b1"
    out.mkdir()
    opened = open_pane(
        world, "--name", "b1", "--keep-on-exit", "never", "--", "sh", str(script), str(out)
    )
    wait_for(lambda: (out / "child.pid").exists() and (out / "leader.pid").exists(), 10, "b1 pids")
    world.track(opened["pid"], tree=True)
    check(
        int((out / "leader.pid").read_text()) == opened["pid"],
        "the pane process is the leader the script recorded",
        "design.md A8",
    )
    killed = world.tmux("kill-pane", "-t", opened["pane_id"])
    check(killed.returncode == 0, "kill-pane works", "design.md B1", killed.stderr)
    wait_for(lambda: (out / "leader-hup").exists(), 10, "SIGHUP at the pane's process")
    time.sleep(1.0)  # the leader is still alive (it sleeps 3 s in its trap)
    check(
        not (out / "child-hup").exists(),
        "while the pane's process lives, its child gets no SIGHUP (spike B1)",
        "design.md B1-B3 / D17",
        "child-hup exists",
    )
    world.observe("b1.child-hup-after-leader-exit", None)


@SC("termctl-keep-on-exit")
def test_keep_on_exit_and_dead_pane_states(world: World) -> None:
    """Spike B6 and promotion checks F/G, #36 #27 (``--keep-on-exit failed``): a failed exit leaves
    a dead pane (``state: dead``, ``exit_status``), a signal death carries ``exit_signal``, a clean
    exit closes the pane (state ``gone``), ``list --dead`` lists only dead panes, ``close`` of a dead
    pane needs no ``--force``."""
    open_pane(world, "--name", "fail", "--keep-on-exit", "failed", "--", "sh", "-c", "exit 3")
    open_pane(world, "--name", "sig", "--keep-on-exit", "failed", "--", "sh", "-c", "kill -9 $$")
    clean = open_pane(
        world, "--name", "clean", "--keep-on-exit", "failed", "--", "sh", "-c", "exit 0"
    )
    open_pane(world, "--name", "always", "--keep-on-exit", "always", "--", "sh", "-c", "exit 0")
    running = open_pane(world, "--name", "run", "--keep-on-exit", "failed", "--", "sleep", "300")
    wait_for(lambda: len(panes(world, "--dead")) >= 3, 15, "dead panes")
    dead = {p["name"]: p for p in panes(world, "--dead")}
    check(
        set(dead) == {"fail", "sig", "always"},
        "list --dead lists the failed, the signaled and the keep-on-exit=always panes only; a clean exit "
        "under 'failed' closes its pane",
        "promotion check F (--keep-on-exit failed keeps only failed panes)",
        sorted(dead),
    )
    check(
        dead["fail"]["state"] == "dead" and dead["fail"]["exit_status"] == 3,
        "a failed exit is state dead with exit_status",
        "#36 #8 termctl exit codes",
        dead["fail"],
    )
    check(
        dead["sig"]["exit_signal"] == 9 and dead["sig"]["exit_status"] is None,
        "a signal death carries exit_signal, no exit_status",
        "#36 #8 termctl exit_signal",
        dead["sig"],
    )
    check(dead["always"]["exit_status"] == 0, "keep-on-exit always keeps a clean exit", "#36 #27")
    shown = world.json_tool("termctl", "show", clean["id"], ok=(0, 10))
    check(
        shown.get("error", {}).get("code") == "not_found" or shown.get("state") == "gone",
        "a clean exit closed the pane",
        "#36 #27: a successful supervisor exit closes the pane",
        shown,
    )
    names = [p["name"] for p in panes(world)]
    check(names.index("run") >= 0, "the running pane is listed", "list")
    world.observe("termctl.list.order", names)
    world.observe("termctl.list --dead.order", [p["name"] for p in panes(world, "--dead")])
    closed = world.json_tool("termctl", "close", dead["fail"]["id"])
    check(
        closed["pane_closed"] is True
        and closed["was_running"] is False
        and closed["forced"] is False,
        "close of a dead pane needs no --force",
        "#36 #79: close of a dead pane without --force",
        closed,
    )
    again = world.tool("termctl", "--json", "close", dead["fail"]["id"])
    check(
        again.returncode == 10,
        "closing it again exits 10 (pane gone)",
        "#36 #27 exit 10 = pane gone",
    )
    refused = world.tool("termctl", "--json", "close", running["id"])
    check(
        refused.returncode == 20,
        "close of a running pane without --force exits 20",
        "#36 #79: exit 20 (running)",
        (refused.returncode, refused.stdout[-200:]),
    )
    forced = world.tool(
        "termctl", "--json", "close", "--force", "--actor", "human:live", running["id"]
    )
    world.observe("termctl.close --force", (forced.returncode, forced.stdout[-300:]))
    check(
        forced.returncode == 0,
        "close --force of a running pane works for a human actor",
        "#36 #79",
        forced.stdout[-200:],
    )


@SC("termctl-default-server")
def test_default_server_selection_and_dead_pane_listing(world: World) -> None:
    """#36 #79 (dead-pane work): ``list --label agent=<id> --dead`` without ``--socket`` selects
    TERMCTL_SOCKET, then TMUX, then ``<TMUX_TMPDIR>/tmux-<uid>/default``; exact label matching;
    every pane carries ``socket_path``; an absent server gives exit 0 with an empty list."""
    empty = world.tool("termctl", "--json", "list", "--label", "agent=ag-none", "--dead")
    world.observe(
        "termctl.list.no-server", (empty.returncode, empty.stdout[-200:], empty.stderr[-200:])
    )
    check(
        empty.returncode == 0,
        "an absent default server gives exit 0 (an empty list)",
        "#36 #79 spot probe (exit 0 with an empty list on a safe path)",
        (empty.returncode, empty.stdout[-200:]),
    )
    for name, label in (("a", "agent=ag-a"), ("ab", "agent=ag-ab")):
        open_pane(
            world,
            "--name",
            name,
            "--label",
            label,
            "--keep-on-exit",
            "failed",
            "--",
            "sh",
            "-c",
            "exit 4",
        )
    wait_for(lambda: len(panes(world, "--dead")) == 2, 15, "two dead panes")
    only = panes(world, "--label", "agent=ag-a", "--dead")
    check(
        [p["name"] for p in only] == ["a"],
        "label filtering is an exact match (agent=ag-a does not match agent=ag-ab)",
        "#36 #79 exact-match label filtering",
        [p["name"] for p in only],
    )
    check(
        all(p.get("socket_path") for p in only),
        "every listed pane carries socket_path",
        "#36 #79 (agent-spawn falls back to the reply's top-level value)",
        only,
    )
    default = world.json_tool("termctl", "list", "--dead")
    check(Path(default["socket_path"]) == world.default_socket, "default server path", "#36 #79")
    # TERMCTL_SOCKET beats TMUX_TMPDIR's default
    other = world.root / "other.sock"
    world.tmux("new-session", "-d", "-s", "x", "sleep 300", socket=other)
    via_env = world.json_tool("termctl", "list", extra={"TERMCTL_SOCKET": str(other)})
    check(
        Path(via_env["socket_path"]) == other and via_env["panes"] == [],
        "TERMCTL_SOCKET selects the server first",
        "#36 #79 order TERMCTL_SOCKET, TMUX, default",
        via_env["socket_path"],
    )
    via_tmux = world.run([str(world.bin / "termctl"), "--json", "list"], tmux=other, timeout=30)
    check(
        via_tmux.returncode == 0 and f'"socket_path": "{other}"' in via_tmux.stdout,
        "TMUX selects the server when TERMCTL_SOCKET is unset",
        "#36 #79 order TERMCTL_SOCKET, TMUX, default",
        via_tmux.stdout[-200:],
    )
    # a dead pane of the default server held by nothing: close it by id
    pane = only[0]
    closed = world.json_tool("termctl", "close", pane["id"])
    check(closed["pane_closed"] is True, "close of a dead pane on the default server", "#36 #79")
    unsafe = world.root / "unsafe"
    unsafe.mkdir()
    unsafe.chmod(0o777)
    bad = world.tool("termctl", "--json", "--socket", str(unsafe / "s"), "list")
    world.observe("termctl.list.unsafe-socket-dir", (bad.returncode, bad.stdout[-200:]))
    check(
        bad.returncode == 12,
        "an unsafe socket directory exits 12",
        "#36 #79 spot probe (exit 12)",
        bad.returncode,
    )


@SC("termctl-focus")
def test_focus_without_a_client_and_unknown_ids(world: World) -> None:
    """#36 #28 and #8 termctl: ``focus`` of a managed pane with no attached client answers
    ``attach`` (exit 0) or fails with exit 12 (``no_client``); an unknown id exits 10; ``--client``
    of an unknown client is an error."""
    opened = open_pane(world, "--name", "f", "--", "sleep", "300")
    done = world.tool("termctl", "--json", "focus", "--no-attach", opened["id"])
    world.observe("termctl.focus --no-attach", (done.returncode, done.stdout[-300:]))
    check(
        done.returncode in (0, 12),
        "focus --no-attach without a client answers (0) or exits 12 no_client",
        "#36 #28 / #8 (real exits 12 without an attached client)",
        (done.returncode, done.stdout[-200:]),
    )
    unknown = world.tool("termctl", "--json", "focus", "tc-nonexist")
    check(unknown.returncode == 10, "unknown id exits 10", "#36 #28", unknown.returncode)
    client = world.tool("termctl", "--json", "focus", "--client", "/dev/pts/9999", opened["id"])
    world.observe("termctl.focus --client unknown", (client.returncode, client.stdout[-300:]))
    check(
        client.returncode in (10, 12),
        "an unknown client is client_not_found/no_client",
        "#36 #28",
        client.returncode,
    )
    labels = world.tool("termctl", "--json", "label", "add", opened["id"], "a=b")
    labels2 = world.tool("termctl", "--json", "label", "add", opened["id"], "c=d")
    check(
        labels.returncode == 0 and labels2.returncode == 0,
        "label add works sequentially",
        "#36 #8 termctl locked",
    )
