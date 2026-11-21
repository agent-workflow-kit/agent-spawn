"""The promotion sub-checks of the issue body (#29) that need no Claude Code.

Real claude-session, termctl and tmux; a stub ``claude``. The checks that need an agent-spawn
promotion window with a real conversation (the promoted run's ``pane_proc`` identity against the
host, continuation under the same Claude session id) are in the billed ``promotion-escalation``.
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import time
from typing import Any

import pytest

from .harness import World, check, wait_for
from .test_claude_session import cs, cs_raw, prepare, start

SC = pytest.mark.live_scenario


def open_exec_pane(world: World, argv: list[str], name: str) -> dict[str, Any]:
    return world.json_tool(
        "termctl",
        "open",
        "--name",
        name,
        "--keep-on-exit",
        "failed",
        "--label",
        f"agent=ag-{name}",
        "--",
        *argv,
    )


@SC("promotion-subchecks")
def test_promotion_checks_against_the_real_programs(world: World) -> None:
    """Promotion checks B, C, D, G (#36 issue body, from PR #73): (B) the real ``status`` of an
    unadopted promoted run: ``pending``, phase ``launch``, ``expires_at``, finalized ``expired`` /
    ``never_adopted`` after the grace; (C) ``start --resume <id> --mode interactive --json`` with
    no prompt file; (D) ``exec --session-id`` inside a real pty (a termctl pane) and its exit 20
    ``expired`` for an unadopted run; (G) the timing of a dead promotion pane appearing in
    ``list --dead`` and its close. Checks A, E and F are in ``termctl-open-pid`` and
    ``termctl-keep-on-exit``."""
    work = prepare(world, "lost_grace: 4s\n")
    world.install_claude(stub_body="exit 0")
    first = start(world, work, "agent=ag-promo")
    ended = world.run(first["argv"], timeout=60)
    check(
        ended.returncode in (0, 14),
        "the first (headless) run finishes",
        "design.md D5",
        ended.stderr[-200:],
    )
    # (C) interactive resume with no prompt file
    resumed = cs(
        world,
        "start",
        "--resume",
        first["csession_id"],
        "--mode",
        "interactive",
        "--cwd",
        str(work),
    )
    check(
        resumed["run"] == 2 and resumed["csession_id"] == first["csession_id"],
        "start --resume --mode interactive makes run 2",
        "promotion check C",
    )
    # (B) status of the unadopted promoted run
    status = cs(world, "status", first["csession_id"])
    run = status["run"]
    check(
        status["pending"] is True and run["phase"] == "launch" and status["expires_at"],
        "an unadopted promoted run is pending, phase launch, with expires_at",
        "promotion check B (#29)",
        {"pending": status["pending"], "phase": run["phase"], "expires_at": status["expires_at"]},
    )
    check(
        run["mode"] == "interactive" and run["run"] == 2,
        "the pending run is run 2, interactive",
        "promotion check C",
    )
    # (D) exec in a real pty: adopt run 2 before it expires, as the host would
    pane = open_exec_pane(world, resumed["argv"], "adopt")
    wait_for(
        lambda: cs(world, "status", first["csession_id"])["run"]["phase"] == "session",
        30,
        "exec to adopt the promoted run inside the pane",
    )
    check(
        pane["pid"]
        == int(
            world.tmux("display-message", "-p", "-t", pane["pane_id"], "#{pane_pid}").stdout.strip()
        ),
        "termctl open's pid is the pane process, which is exec itself when exec is the pane command",
        "promotion check A (pane PID) / #36 #29",
    )
    cs(world, "stop", first["csession_id"])
    wait_for(
        lambda: not cs(world, "status", first["csession_id"])["live"], 30, "the stopped run to end"
    )
    # an unadopted run that expires, then exec of it
    third = cs(
        world,
        "start",
        "--resume",
        first["csession_id"],
        "--mode",
        "interactive",
        "--cwd",
        str(work),
    )
    check(third["run"] == 3, "resume again makes run 3", "design.md Resume order")
    time.sleep(5.5)  # lost_grace 4s
    final = cs(world, "status", first["csession_id"])["run"]
    check(
        final["outcome"] == "not-started" and final["end_reason"] == "never_adopted",
        "the unadopted promoted run finalizes not-started / never_adopted",
        "promotion check B (finalization)",
        {k: final.get(k) for k in ("outcome", "end_reason")},
    )
    open_exec_pane(world, third["argv"], "late")
    began = time.monotonic()
    dead = wait_for(
        lambda: [
            p
            for p in world.json_tool("termctl", "list", "--label", "agent=ag-late", "--dead")[
                "panes"
            ]
        ],
        20,
        "the dead pane of the expired exec",
    )
    appeared = time.monotonic() - began
    world.observe("termctl.dead-pane.appeared-after-s", round(appeared, 2))
    check(
        dead[0]["exit_status"] == 20,
        "exec --session-id of an expired run exits 20 inside the pane (the host must still exit 0)",
        "promotion check D (exit 20 expired for an unadopted run)",
        dead[0],
    )
    check(
        appeared < 10,
        "a dead promotion pane shows in list --dead within seconds of its death (G)",
        "promotion check G",
        appeared,
    )
    closed = world.json_tool("termctl", "close", dead[0]["id"])
    check(
        closed["pane_closed"] is True,
        "close_dead of the dead promotion pane",
        "promotion check G",
        closed,
    )
    out = cs_raw(world, "--json", "status", first["csession_id"])
    check(out.returncode == 0, "the session stays readable", "design.md D47")
