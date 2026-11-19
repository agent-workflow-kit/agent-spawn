"""agent-spawn's operational commands against real state: cron, reconcile --reset-lease, browse.

No Claude Code and no real crontab: ``cron`` runs against a fake ``crontab`` that is first on the
private ``PATH`` (the table is a file in the private root; the tripwire stub that is otherwise
first on ``PATH`` fails any other ``crontab`` call with exit 70). Real-crontab fidelity (the exact
``no crontab for <user>`` wording and exit code of the system's cron, localized messages) is NOT
checked on purpose and is listed in docs/live-testing.md.
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from .harness import World, check, write_fake_crontab
from .test_agent_spawn_stub import stub_setup

SC = pytest.mark.live_scenario

BEGIN = "# agent-spawn managed"
END = "# agent-spawn managed end"


@SC("as-cron-fake-crontab")
def test_cron_install_replace_and_uninstall_against_a_fake_crontab(world: World) -> None:
    """#36 #34 and FIDELITY.md crontab: the block is read with ``crontab -l`` and written with
    ``crontab -`` on stdin; install appends, a second install is ``unchanged``; every other line
    of the table is kept; uninstall removes the block and, with nothing else left, runs
    ``crontab -r``. A failing crontab exits 12 and writes nothing."""
    # the tripwire first: a crontab that fails
    failing = world.agent_spawn("cron", "install", "--json")
    check(
        failing.returncode == 12 and not world.crontab_table.exists(),
        "a crontab that fails to read exits 12 and writes nothing",
        "interface.md cron: any other failure to read or write exits 12",
        (failing.returncode, failing.stdout[-200:]),
    )
    write_fake_crontab(world.bin / "crontab", world.crontab_table)
    world.crontab_table.write_text("MAILTO=nobody@example.invalid\n0 3 * * * /usr/bin/true\n")
    installed = world.spawn_json("cron", "install", "--interval", "7")
    world.observe("agent-spawn.cron.install", installed)
    table = world.crontab_table.read_text()
    lines = table.splitlines()
    check(
        lines[:2] == ["MAILTO=nobody@example.invalid", "0 3 * * * /usr/bin/true"],
        "every other line of the table is kept as it was",
        "interface.md cron (Table handling)",
        lines[:2],
    )
    check(
        BEGIN in lines and END in lines,
        "the managed block markers are written",
        "interface.md cron (Block)",
    )
    job = lines[lines.index(BEGIN) + 1]
    check(
        job.startswith("*/7 * * * * ")
        and job.endswith("reconcile --quiet")
        and f"'{world.agent_spawn_path}'" in job
        and "PATH='" in job
        and f"XDG_STATE_HOME='{world.state}'" in job,
        "the job runs `<abs agent-spawn> reconcile --quiet` every 7 minutes with an explicit PATH and "
        "the installing environment's XDG_STATE_HOME",
        "interface.md cron (Block)",
        job,
    )
    again = world.spawn_json("cron", "install", "--interval", "7")
    check(
        json.dumps(again).count("unchanged") >= 1,
        "installing the same block again reports unchanged",
        "interface.md cron (install replaces in place, unchanged when the same)",
        again,
    )
    check(
        world.crontab_table.read_text() == table,
        "an unchanged install writes nothing",
        "interface.md cron",
    )
    replaced = world.spawn_json("cron", "install", "--interval", "9")
    check(
        world.crontab_table.read_text().splitlines().count(BEGIN) == 1
        and "*/9 * * * * " in world.crontab_table.read_text(),
        "a new interval replaces the block in place (exactly one block)",
        "interface.md cron",
        replaced,
    )
    gone = world.spawn_json("cron", "uninstall")
    world.observe("agent-spawn.cron.uninstall", gone)
    left = world.crontab_table.read_text()
    check(
        BEGIN not in left and "0 3 * * * /usr/bin/true" in left,
        "uninstall removes only the block",
        "interface.md cron",
    )
    # a foreign job between the markers is refused
    world.crontab_table.write_text(f"{BEGIN}\n*/10 * * * * foreign-job\n{END}\n")
    refused = world.agent_spawn("cron", "uninstall", "--json")
    check(
        refused.returncode == 12 and "foreign-job" in world.crontab_table.read_text(),
        "a foreign job inside the markers exits 12 and nothing is written",
        "interface.md cron (Block)",
        (refused.returncode, refused.stdout[-200:]),
    )
    world.crontab_table.write_text("")
    world.agent_spawn("cron", "install", "--json")
    world.spawn_json("cron", "uninstall")
    check(
        not world.crontab_table.exists(),
        "with nothing but blank lines left uninstall runs `crontab -r`",
        "interface.md cron (Table handling)",
    )


def corrupt_lease(db: Path, holder: Any) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO leases (key, holder, acquired) VALUES ('reconcile', ?, 0)",
            (holder,),
        )
        conn.commit()
    finally:
        conn.close()


@SC("as-reconcile-reset-lease")
def test_reset_lease_repairs_a_corrupted_state_db_row(world: World) -> None:
    """#36 #80 (a real ``reconcile --reset-lease`` against a corrupted ``state.db`` row: invalid
    UTF-8 bytes, a blob, deeply nested JSON; human actor; the warning when ``log.jsonl`` cannot be
    written) and design.md D52."""
    stub_setup(world, "exit 0")
    launched = world.spawn_json(
        "launch",
        "--mode",
        "headless",
        "--prompt",
        "reply ok",
        "--model",
        "haiku",
        "--wait",
        ok=(0, 14),
    )
    check(launched["agent"]["agent_id"], "a launch creates state.db", "design.md D52")
    db = world.state / "agent-spawn" / "state.db"
    check(db.exists(), "state.db exists under the private XDG_STATE_HOME", "interface.md Files", db)
    log = world.state / "agent-spawn" / "log.jsonl"
    holders: list[tuple[str, Any]] = [
        ("invalid UTF-8 bytes", b"\xff\xfe\x00\xc3("),
        ("deeply nested JSON", "[" * 200000 + "]" * 200000),
        ("plain garbage text", "not a holder"),
    ]
    for label, holder in holders:
        corrupt_lease(db, holder)
        plain = world.agent_spawn("reconcile", "--json")
        err = json.loads(plain.stdout)["error"]
        check(
            plain.returncode == 12 and "--reset-lease" in json.dumps(err),
            f"a plain reconcile over a {label} holder exits 12 and names --reset-lease",
            "design.md D52",
            (plain.returncode, plain.stdout[-300:]),
        )
        dry = world.agent_spawn(
            "reconcile", "--reset-lease", "--dry-run", "--actor", "human:live", "--json"
        )
        check(
            dry.returncode == 2,
            "--reset-lease with --dry-run exits 2",
            "design.md D52",
            dry.returncode,
        )
        auto = world.agent_spawn("reconcile", "--reset-lease", "--actor", "auto:x", "--json")
        check(
            auto.returncode == 2,
            "--reset-lease is human-only (an auto: actor exits 2)",
            "design.md D52",
            auto.stdout[-200:],
        )
        repaired = world.agent_spawn(
            "reconcile", "--reset-lease", "--actor", "human:live", "--json"
        )
        out = json.loads(repaired.stdout)
        check(
            repaired.returncode == 0
            and any("could not be parsed and was replaced" in w for w in out["warnings"]),
            f"--reset-lease as a human replaces a {label} holder and warns",
            "#36 #80 (real corrupted state.db row)",
            (repaired.returncode, repaired.stdout[-300:]),
        )
        again = world.agent_spawn("reconcile", "--json")
        check(
            again.returncode == 0,
            "the next plain reconcile works",
            "design.md D52",
            again.stdout[-200:],
        )
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    resets = [r for r in records if r.get("details", {}).get("kind") == "lease_reset"]
    check(
        len(resets) == len(holders),
        "each repair is audited in log.jsonl",
        "design.md D52",
        len(resets),
    )
    world.observe("agent-spawn.lease-reset.log-record.keys", sorted(resets[0]))
    # an unwritable log.jsonl: the warning says the record could not be written
    corrupt_lease(db, "not a holder")
    log.chmod(0o444)
    state_dir = log.parent
    state_dir.chmod(0o500)
    try:
        quiet = world.agent_spawn("reconcile", "--reset-lease", "--actor", "human:live", "--json")
    finally:
        state_dir.chmod(0o700)
        log.chmod(0o600)
    world.observe("agent-spawn.lease-reset.unwritable-log", (quiet.returncode, quiet.stdout[-300:]))
    check(
        quiet.returncode == 0
        and any(
            "the log record could not be written" in w for w in json.loads(quiet.stdout)["warnings"]
        ),
        "an unwritable log.jsonl ends the warning with '(the log record could not be written)'",
        "#36 #80",
        (quiet.returncode, quiet.stdout[-300:]),
    )


@SC("as-browse-pilot")
def test_browse_lists_and_shows_real_agents(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """#36 #38 (browse): the Textual pilot against the real ``agent-spawn list`` payload and
    ``show`` output (the payload the fakes and the unit tests only imitate)."""
    import asyncio

    from textual.widgets import DataTable, Static

    from agent_spawn.tui.app import BrowseApp
    from agent_spawn.tui.model import SubprocessBackend

    stub_setup(world, "exit 0")
    ids = []
    for _ in range(2):
        done = world.spawn_json(
            "launch",
            "--mode",
            "headless",
            "--prompt",
            "reply ok",
            "--model",
            "haiku",
            "--wait",
            ok=(0, 14),
        )
        ids.append(done["agent"]["agent_id"])
    for name, value in world.env().items():
        monkeypatch.setenv(name, value)
    backend = SubprocessBackend(
        config_path=None, actor="human:live", prefix=[world.agent_spawn_path]
    )
    app = BrowseApp(backend)
    seen: dict[str, Any] = {}

    async def script() -> None:
        async with app.run_test(size=(140, 30)) as pilot:
            for _ in range(10):
                await app.workers.wait_for_complete()
                await pilot.pause()
                if not app.workers:
                    break
            table = app.query_one("#table", DataTable)
            seen["rows"] = sorted(str(k.value) for k in table.rows)
            await pilot.press("enter")
            for _ in range(10):
                await app.workers.wait_for_complete()
                await pilot.pause()
                if not app.workers:
                    break
            seen["detail"] = str(app.query_one("#detail-text", Static).render())

    try:
        asyncio.run(script())
    finally:
        backend.close()
    check(
        seen["rows"] == sorted(ids),
        "browse shows every agent of the real list payload",
        "#36 #38 browse against real list output",
        seen["rows"],
    )
    check(
        "unreported" in seen["detail"] or "parked" in seen["detail"],
        "enter loads the real show output into the detail pane",
        "#36 #38 browse",
        seen["detail"][:200],
    )
    world.observe("browse.detail.excerpt", seen["detail"][:300])
