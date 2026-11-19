"""Billed scenarios, wave 2: resume, fork, teardown and promotion with real Claude Code.

See ``test_billed_basic.py`` for the conventions; these tests have not been run against a real
Claude Code yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .billed import (
    bsetup,
    cs_status,
    expect_parked,
    launch_headless,
    show,
    spawn,
    spawn_raw,
    wait_agent,
)
from .harness import World, check

SC = pytest.mark.live_scenario


@SC("resume-continue")
def test_headless_resume_continues_the_claude_session(billed_world: World) -> None:
    """Run 1 asks claude to remember a word; ``resume`` (run 2) continues the same Claude
    conversation: the Claude session id does not change, ``list --label agent=<id>`` lists the
    session once, and the run counters advance. Confirms #36 #75 5 (resume twice lists once),
    design.md D5 (continuation under the same Claude session id) and the resume order."""
    world = billed_world
    wt = bsetup(world)
    first = launch_headless(
        world,
        wt,
        "Remember the word pear. Reply with the single word ok.",
        "--worktree",
        "none",
        "--cwd",
        str(wt.repo.clone),
    )
    agent = first["agent"]
    aid = agent["agent_id"]
    before = cs_status(world, agent["csession_id"])
    second = spawn(
        world,
        "resume",
        aid,
        "--mode",
        "headless",
        "--model",
        "haiku",
        "--max-turns",
        "3",
        "--prompt",
        "What word did I ask you to remember? Reply with that word only.",
        "--wait",
    )
    check(
        second["agent"]["run"] == 2 and second["outcome"] == "done",
        "resume makes run 2 and it is done",
        "interface.md Resume order",
        {k: second[k] for k in ("outcome", "end_reason")},
    )
    after = cs_status(world, agent["csession_id"])
    check(
        after["claude_session_ids"] == before["claude_session_ids"]
        and after["claude_session_id"] == before["claude_session_id"],
        "resume continues the same Claude session (the id does not change)",
        "design.md D5 / claude-session resume",
        (before["claude_session_ids"], after["claude_session_ids"]),
    )
    listed = world.json_tool("claude-session", "list", "--label", f"agent={aid}")["sessions"]
    check(
        len(listed) == 1,
        "after two runs list --label agent=<id> lists the session once",
        "#36 #75 5",
        len(listed),
    )
    world.observe("resume.run2.num_turns", after["run"].get("num_turns"))
    world.spawn_json("teardown", aid)


@SC("fork-inplace-holder")
def test_in_place_fork_holds_the_worktree_and_the_original_is_held(billed_world: World) -> None:
    """A real run in a ``--worktree new`` worktree, then ``resume --fork`` in place: the fork is a
    new agent with its own Claude session that holds the same wtctl record (holders sorted); the
    original's teardown is refused (``held``) until the fork is torn down; the fork must not
    inherit the source's ``agent=`` label. Confirms #36 #21, #22, #23, #75 5 and the in-place fork
    holder of the issue."""
    world = billed_world
    wt = bsetup(world)
    first = launch_headless(
        world,
        wt,
        "Remember the word pear. Reply with the single word ok.",
        "--repo",
        wt.repo.repo_id,
        "--worktree",
        "new",
    )
    agent = first["agent"]
    aid = agent["agent_id"]
    record = wt.ok("list")["records"][0]
    check(aid in record["holders"], "the launch holds its worktree record", "design.md D6")
    forked = spawn(
        world,
        "resume",
        aid,
        "--fork",
        "--mode",
        "headless",
        "--model",
        "haiku",
        "--max-turns",
        "3",
        "--prompt",
        "What word did I ask you to remember? Reply with that word only.",
        "--wait",
    )
    fork = forked["agent"]
    fid = fork["agent_id"]
    check(
        fork["forked_from"] == aid and fork["csession_id"] != agent["csession_id"],
        "the fork is a new agent with its own session",
        "design.md D25/D46",
    )
    source_label = world.json_tool("claude-session", "list", "--label", f"agent={aid}")["sessions"]
    check(
        [s["csession_id"] for s in source_label] == [agent["csession_id"]],
        "list --label agent=<source id> returns exactly the source session after a fork",
        "#36 #75 5 (fork label inheritance)",
        [s["labels"] for s in source_label],
    )
    check(
        wt.ok("show", record["id"])["holders"] == sorted([aid, fid]),
        "the fork holds the same record; holders are sorted",
        "#36 #21 #22",
    )
    refused = spawn_raw(world, "teardown", aid)
    check(
        refused.returncode == 20 and "held" in refused.stdout,
        "the original's teardown is refused while the fork holds the worktree",
        "#36 #22 #23 (held)",
        (refused.returncode, refused.stdout[-200:]),
    )
    check(Path(record["path"]).exists(), "the worktree is untouched", "design.md D14")
    world.spawn_json("teardown", fid)
    world.spawn_json("teardown", aid)
    check(
        wt.ok("show", record["id"])["state"] == "removed" and not Path(record["path"]).exists(),
        "after both teardowns the real wtctl removed the worktree",
        "#36 #23",
    )


@SC("fork-throwaway")
def test_fork_into_a_throwaway(billed_world: World) -> None:
    """A third-party-reading run in a wtctl throwaway, then ``resume --fork --cwd <other
    throwaway>``: throwaways are never held, the fork runs in the other throwaway's root, and the
    throwaway records survive both teardowns. Confirms #36 #21 and design.md D6, D49 (forks into
    throwaways). Skips (recorded) when a third-party-reading run cannot start on this host (no
    sandbox)."""
    world = billed_world
    wt = bsetup(world)
    ref = wt.repo.sha
    first_record = wt.ok("create", "--throwaway", "--repo", wt.repo.repo_id, "--ref", ref)["record"]
    second_record = wt.ok("create", "--throwaway", "--repo", wt.repo.repo_id, "--ref", ref)[
        "record"
    ]
    flags = ["--reads-third-party", "--sandbox", "preferred", "--cwd", first_record["path"]]
    dry = world.agent_spawn(
        "launch",
        "--dry-run",
        "--mode",
        "headless",
        "--prompt",
        "x",
        "--model",
        "haiku",
        *flags,
        "--json",
        auth=False,
    )
    if dry.returncode != 0:
        pytest.skip(
            f"fork-throwaway: a third-party-reading run cannot start here: {dry.stdout[-300:]}"
        )
    world.trust(Path(first_record["path"]), Path(second_record["path"]), wt.repo.clone)
    first = launch_headless(
        world, wt, "Remember the word pear. Reply with the single word ok.", *flags
    )
    agent = first["agent"]
    aid = agent["agent_id"]
    check(
        agent["cwd"] == first_record["path"] and not agent["holds"],
        "a throwaway is never held",
        "design.md D6",
    )
    forked = spawn(
        world,
        "resume",
        aid,
        "--fork",
        "--cwd",
        second_record["path"],
        "--mode",
        "headless",
        "--model",
        "haiku",
        "--max-turns",
        "3",
        "--prompt",
        "Reply with the single word ok.",
        "--wait",
    )
    fork = forked["agent"]
    check(
        fork["cwd"] == second_record["path"] and fork["forked_from"] == aid and not fork["holds"],
        "the fork runs in the other throwaway's root and holds nothing",
        "design.md D25/D49 (fork --cwd into a throwaway)",
        {k: fork[k] for k in ("cwd", "forked_from", "holds")},
    )
    for agent_id in (fork["agent_id"], aid):
        world.spawn_json("teardown", agent_id)
    kept = {r["id"] for r in wt.ok("list")["records"]}
    check(
        {first_record["id"], second_record["id"]} <= kept,
        "teardown never removes a throwaway record (they are not held)",
        "design.md D14 / D6",
        kept,
    )


@SC("teardown-wtctl-real")
def test_teardown_of_a_launched_worktree_with_the_real_wtctl(billed_world: World) -> None:
    """``launch --worktree new`` with a real run, then ``teardown``: wtctl ``show`` before, the
    owned record removed through ``wtctl remove --yes --holder`` as an ``auto:`` actor without a
    TTY, the session forgotten, the agent a tombstone. Confirms #36 #20, #23 and the issue's
    teardown with real wtctl."""
    world = billed_world
    wt = bsetup(world)
    done = launch_headless(world, wt, PROMPT, "--repo", wt.repo.repo_id, "--worktree", "new")
    agent = done["agent"]
    aid = agent["agent_id"]
    shown = show(world, aid)
    check(
        shown["repo_config_changed"] is not None,
        "show reports the baseline check of its own record",
        "design.md D49 / #36 #25",
    )
    record = wt.ok("list")["records"][0]
    world.observe("teardown.show.repo_config_changed", shown["repo_config_changed"])
    torn = world.spawn_json("teardown", aid)
    gone = wt.ok("show", record["id"])
    check(
        gone["state"] == "removed" and not Path(record["path"]).exists(),
        "teardown removed the owned worktree through the real wtctl",
        "#36 #23",
        gone["state"],
    )
    check(
        torn["session_forgotten"] is True
        and world.tool("claude-session", "status", agent["csession_id"], "--json").returncode == 10,
        "teardown forgot the session",
        "design.md D14",
        torn,
    )
    check(
        world.agent_spawn("show", aid, "--json").returncode in (0, 10),
        "the agent is a tombstone",
        "design.md D14",
    )


PROMPT = "Reply with the single word ok."


@SC("promotion-escalation")
def test_headless_needs_input_is_promoted_into_a_window_and_continues(billed_world: World) -> None:
    """Spike D5 through agent-spawn: a headless run reports ``needs-input``; promotion (``kinds.
    implement.promote``) opens a window whose ``supervise --host`` resumes the same Claude session
    interactively (the conversation continues under the same Claude session id); the host's pane
    process matches the recorded host identity; closing the window parks the agent. Confirms
    #36 #29 and #73 (pane pid under the real termctl, an unadopted promoted run's status),
    design.md D5, D30, D47."""
    world = billed_world
    wt = bsetup(world, spawn_extra="kinds:\n  implement:\n    promote: true\n")
    prompt = (
        "Run exactly this shell command, then stop and say nothing else: "
        'claude-session report needs-input --reason="need a decision"'
    )
    launched = spawn(
        world,
        "launch",
        "--mode",
        "headless",
        "--prompt",
        prompt,
        "--model",
        "haiku",
        "--max-turns",
        "4",
        "--timeout",
        "5m",
        "--worktree",
        "none",
        "--cwd",
        str(wt.repo.clone),
        "--promote",
        ok=(0, 16, 17, 18),
    )
    aid = launched["agent"]["agent_id"]
    state = wait_agent(
        world,
        aid,
        lambda s: (
            s["agent"].get("promotion_pending")
            or len(s["runs"]) >= 2
            or s["agent"]["status"] == "parked"
        ),
        "the promotion to start or the agent to park",
        180,
    )
    world.observe(
        "promotion.after-run1",
        {
            k: state["agent"][k]
            for k in ("status", "park_reason", "needs_attention", "promotion_pending")
        },
    )
    runs = wait_agent(world, aid, lambda s: len(s["runs"]) >= 2, "the promoted run (run 2)", 120)[
        "runs"
    ]
    check(
        runs[1]["promoted"] is True and runs[1]["mode"] == "interactive",
        "the second run is the promoted interactive run",
        "interface.md Promotion / #36 #29",
        runs[1],
    )
    panes = world.json_tool("termctl", "list", "--label", f"agent={aid}")["panes"]
    check(len(panes) == 1, "the promotion opened one window", "design.md D30", panes)
    host = wait_agent(
        world, aid, lambda s: s["agent"]["status"] == "live", "the promoted run to start", 120
    )
    check(
        host["supervisor"]["pid"] == panes[0]["pid"],
        "the host's pane process equals the recorded supervisor (pane_proc identity)",
        "#36 #29 (A) pane PID under tmux's default-shell wrapping",
        (host["supervisor"], panes[0]["pid"]),
    )
    status = cs_status(world, launched["agent"]["csession_id"])
    check(
        len(status["claude_session_ids"]) == 1,
        "the promoted run continues the same Claude session id",
        "design.md D5",
        status["claude_session_ids"],
    )
    world.track(panes[0]["pid"], tree=True)
    world.tmux("kill-window", "-t", panes[0]["window_id"])
    parked = expect_parked(world, aid, 90)
    world.observe(
        "promotion.final",
        {k: parked[k] for k in ("park_reason", "needs_attention", "last_outcome")},
    )
    world.spawn_json("teardown", aid, ok=(0, 20))
