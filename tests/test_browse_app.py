"""Pilot tests of the browse screen: every key, against a scripted backend."""

from __future__ import annotations

import json
from typing import Any

from textual.coordinate import Coordinate
from textual.widgets import DataTable, Input, Static

from agent_spawn.tui.app import BrowseApp, ConfirmScreen, HelpScreen
from agent_spawn.tui.model import CliResult
from tests import schema_registry
from tests.browse_support import IDS, NOW, FakeBackend, agent, run_app, settle

HOSTILE = "[bold red]x[/] \x1b[2J\x1b]0;t\x07\x00 [/]"


def three() -> list[dict[str, Any]]:
    return [
        agent(IDS[0], status="live", labels=["ticket=A-1"], needs_attention=None),
        agent(IDS[1], status="parked", labels=["ticket=B-2"], needs_attention="blocked"),
        agent(IDS[2], status="parked", labels=["owner=zed"]),
    ]


def make(backend: FakeBackend, **kwargs: Any) -> BrowseApp:
    return BrowseApp(backend, clock=lambda: NOW, **kwargs)


def text_of(app: BrowseApp, selector: str) -> str:
    return str(app.query_one(selector, Static).render())


def row_ids(app: BrowseApp) -> list[str]:
    return [str(k.value) for k in app.query_one("#table", DataTable).rows]


def test_lists_agents_and_loads_nothing_but_list_at_start() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        assert row_ids(app) == IDS
        assert backend.commands() == [["list"]]
        assert "press enter" in text_of(app, "#detail-text")

    run_app(make(backend), script)


def test_enter_loads_show_into_the_detail_pane() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("down", "enter")
        await settle(app, pilot)
        assert ["show", IDS[1]] in backend.commands()
        assert "blocked" in text_of(app, "#detail-text")
        assert "press enter" not in text_of(app, "#detail-text")

    run_app(make(backend), script)


def test_show_failure_is_shown_in_the_detail_pane() -> None:
    backend = FakeBackend(
        three(), show=CliResult(12, {"error": {"code": "failed", "message": "unreadable [x]"}})
    )

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("enter")
        await settle(app, pilot)
        assert "show failed: exit 12: unreadable [x]" in text_of(app, "#detail-text")

    run_app(make(backend), script)


def test_focus_opens_on_that_agent_and_loads_it() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        assert app.query_one("#table", DataTable).cursor_row == 2
        assert ["show", IDS[2]] in backend.commands()

    run_app(make(backend, focus="ag-cccc"), script)


def test_focus_on_unknown_agent_keeps_the_list_and_says_so() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        assert row_ids(app) == IDS
        assert "no listed agent matches" in text_of(app, "#status")

    run_app(make(backend, focus="ag-zzzz"), script)


def test_refresh_lists_again_and_keeps_the_cursor_agent() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("down")
        backend.agents = [agent("ag-dddddddd"), *backend.agents]
        await pilot.press("R")
        await settle(app, pilot)
        assert backend.commands().count(["list"]) == 2
        assert row_ids(app)[0] == "ag-dddddddd"
        assert app.query_one("#table", DataTable).cursor_row == 2  # still on IDS[1]

    run_app(make(backend), script)


def test_label_filter_narrows_and_escape_clears() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("slash")
        assert app.focused is app.query_one("#filter", Input)
        await pilot.press(*"owner")
        await settle(app, pilot)
        assert row_ids(app) == [IDS[2]]
        await pilot.press("enter")  # keeps the filter, back to the table
        assert app.focused is app.query_one("#table", DataTable)
        assert row_ids(app) == [IDS[2]]
        await pilot.press("escape")
        await settle(app, pilot)
        assert row_ids(app) == IDS

    run_app(make(backend), script)


def test_escape_in_the_filter_box_clears_it() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("slash", *"zed", "escape")
        await settle(app, pilot)
        assert row_ids(app) == IDS and app.query_one("#filter", Input).value == ""

    run_app(make(backend), script)


def test_status_and_attention_filters() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("f")  # launching
        assert "No agent matches" in text_of(app, "#empty")
        await pilot.press("f")  # live
        assert row_ids(app) == [IDS[0]]
        await pilot.press("f")  # parked
        assert row_ids(app) == IDS[1:]
        await pilot.press("n")
        assert row_ids(app) == [IDS[1]]
        assert "attention" in text_of(app, "#filters")

    run_app(make(backend), script)


def test_empty_and_error_states() -> None:
    empty = FakeBackend([])

    async def check_empty(app: BrowseApp, pilot: Any) -> None:
        assert "No agents yet" in text_of(app, "#empty")

    run_app(make(empty), check_empty)

    failing = FakeBackend([])
    failing.list_result = CliResult(
        12, {"error": {"code": "failed", "message": "state.db is busy"}}
    )

    async def check_error(app: BrowseApp, pilot: Any) -> None:
        assert "Could not list agents: exit 12: state.db is busy" in text_of(app, "#empty")
        await pilot.press("a", "k", "enter")  # no agent: nothing runs
        assert failing.commands() == [["list"]]

    run_app(make(failing), check_error)


def test_ack_runs_the_cli_command_as_a_changing_call() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("down", "k")
        await settle(app, pilot)
        assert (["ack", IDS[1]], True) in backend.calls
        assert "ack " + IDS[1] + ": ok" in text_of(app, "#status")
        assert backend.commands()[-1] == ["list"]  # refreshed afterwards

    run_app(make(backend), script)


def test_error_of_an_action_is_shown() -> None:
    backend = FakeBackend(
        three(), ack=CliResult(20, {"error": {"code": "refused", "message": "no"}})
    )

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("k")
        await settle(app, pilot)
        assert f"ack {IDS[0]}: exit 20: no" in text_of(app, "#status")

    run_app(make(backend), script)


def test_stop_asks_first_and_runs_on_yes() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("s")
        assert isinstance(app.screen, ConfirmScreen)
        assert ["stop", IDS[0], "--yes"] not in backend.commands()
        await pilot.press("y")
        await settle(app, pilot)
        assert (["stop", IDS[0], "--yes"], True) in backend.calls

    run_app(make(backend), script)


def test_stop_cancelled_by_no_escape_and_the_default_button() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        for keys in (["n"], ["escape"], ["enter"]):  # enter presses the focused default: No
            await pilot.press("s", *keys)
            await settle(app, pilot)
            assert not isinstance(app.screen, ConfirmScreen)
        assert backend.commands() == [["list"]]
        assert "stop cancelled" in text_of(app, "#status")

    run_app(make(backend), script)


def test_teardown_confirmation_then_yes_flag() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("down", "t")
        await pilot.press("n")
        await settle(app, pilot)
        assert not any(c[0] == "teardown" for c in backend.commands())
        await pilot.press("t", "y")
        await settle(app, pilot)
        assert (["teardown", IDS[1], "--yes"], True) in backend.calls

    run_app(make(backend), script)


def test_confirmation_text_is_plain_for_hostile_names() -> None:
    backend = FakeBackend([agent(IDS[0], name=HOSTILE)])

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("t")
        question = str(app.screen.query_one("#question", Static).render())
        assert "[bold red]x[/]" in question and "\x1b" not in question
        await pilot.press("n")

    run_app(make(backend), script)


def test_wait_reports_the_outcome_without_blocking_other_keys() -> None:
    result = {"agent": None, "phase": "ended", "outcome": "done", "timed_out": False}
    backend = FakeBackend(three(), wait=CliResult(0, result))

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("w")
        await settle(app, pilot)
        assert ["wait", IDS[0], "--timeout", "60s"] in backend.commands()
        assert "outcome=done" in text_of(app, "#status")

    run_app(make(backend), script)


def test_wait_timeout_and_error() -> None:
    timed = {"agent": None, "phase": None, "outcome": None, "timed_out": True}
    backend = FakeBackend(three(), wait=CliResult(0, timed))

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("w")
        await settle(app, pilot)
        assert "still running" in text_of(app, "#status")

    run_app(make(backend), script)
    backend = FakeBackend(
        three(), wait=CliResult(10, {"error": {"code": "not_found", "message": "gone"}})
    )

    async def script2(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("w")
        await settle(app, pilot)
        assert "exit 10: gone" in text_of(app, "#status")

    run_app(make(backend), script2)


def test_resume_is_interactive_in_a_window() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("down", "r")
        await settle(app, pilot)
        expected = ["resume", IDS[1], "--mode", "interactive", "--placement", "window"]
        assert (expected, True) in backend.calls
        assert app.intent is None

    run_app(make(backend), script)


def test_resume_of_marked_agents_runs_each_and_reports_failures() -> None:
    backend = FakeBackend(
        three(), resume=CliResult(20, {"error": {"code": "refused", "message": "live"}})
    )

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("space", "down", "down", "space", "r")
        await settle(app, pilot)
        resumed = [c[1] for c in backend.commands() if c[0] == "resume"]
        assert resumed == [IDS[0], IDS[2]]
        assert "exit 20: live" in text_of(app, "#status")

    run_app(make(backend), script)


def test_attach_runs_attach_then_hands_the_terminal_to_its_argv() -> None:
    answer = {"agent_id": IDS[0], "action": "attach", "argv": ["tmux", "attach", "-t", "x"]}
    backend = FakeBackend(three(), attach=CliResult(0, answer))
    app = make(backend)

    async def on_terminal(argv: list[str]) -> int:
        return backend.run_on_terminal(argv)

    app._on_terminal = on_terminal  # type: ignore[method-assign]  # the headless driver cannot suspend

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("a")
        await settle(app, pilot)
        assert (["attach", IDS[0]], True) in backend.calls
        assert backend.terminal == [["tmux", "attach", "-t", "x"]]

    run_app(app, script)


def test_attach_switched_does_not_touch_the_terminal() -> None:
    answer = {"agent_id": IDS[0], "action": "switched", "argv": ["tmux", "switch-client"]}
    backend = FakeBackend(three(), attach=CliResult(0, answer))

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("a")
        await settle(app, pilot)
        assert backend.terminal == [] and "switched" in text_of(app, "#status")

    run_app(make(backend), script)


def test_attach_failure_and_multiple_marks() -> None:
    backend = FakeBackend(
        three(), attach=CliResult(10, {"error": {"code": "not_found", "message": "not live"}})
    )

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("a")
        await settle(app, pilot)
        assert "exit 10: not live" in text_of(app, "#status")
        await pilot.press("space", "down", "space", "a")
        assert "attach takes one agent" in text_of(app, "#status")
        assert [c for c in backend.commands() if c[0] == "attach"] == [["attach", IDS[0]]]

    run_app(make(backend), script)


def test_a_second_action_while_busy_is_refused() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        app._busy = "something"
        await pilot.press("k")
        assert "busy: something" in text_of(app, "#status")
        assert not any(c[0] == "ack" for c in backend.commands())
        app._busy = None

    run_app(make(backend), script)


def test_help_and_quit() -> None:
    backend = FakeBackend(three())

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("question_mark")
        assert isinstance(app.screen, HelpScreen)
        await pilot.press("escape")
        assert not isinstance(app.screen, HelpScreen)
        await pilot.press("q")

    app = make(backend)
    run_app(app, script)
    assert app.intent is None and backend.closed


def test_intent_mode_attach_records_and_exits_without_running_attach() -> None:
    backend = FakeBackend(three())
    app = make(backend, record_intent=True)

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("down", "a")

    run_app(app, script)
    assert app.intent == ("attach", [IDS[1]])
    assert not any(c[0] == "attach" for c in backend.commands())


def test_intent_mode_resume_carries_the_marked_agents_in_list_order() -> None:
    backend = FakeBackend(three())
    app = make(backend, record_intent=True)

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("down", "down", "space", "up", "up", "space", "r")

    run_app(app, script)
    assert app.intent == ("resume", [IDS[0], IDS[2]])
    assert not any(c[0] == "resume" for c in backend.commands())


def test_intent_mode_quit_is_back_and_empty_list_records_nothing_to_attach() -> None:
    backend = FakeBackend(three())
    app = make(backend, record_intent=True)

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("q")

    run_app(app, script)
    assert app.intent == ("back", [])
    empty = make(FakeBackend([]), record_intent=True)

    async def script2(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("a", "r")
        assert app.intent is None  # nothing to select: stays open

    run_app(empty, script2)


def test_untrusted_strings_render_literally_and_without_control_characters() -> None:
    hostile = agent(
        IDS[0], name=HOSTILE, kind=HOSTILE, labels=[HOSTILE, "[/]"], needs_attention=None
    )
    backend = FakeBackend(
        [hostile],
        show=CliResult(
            0,
            {
                "agent": hostile,
                "runs": [{"run": 1, "mode": HOSTILE}],
                "session": None,
                "supervisor": None,
                "warnings": [HOSTILE],
            },
        ),
    )

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("enter")
        await settle(app, pilot)
        detail = text_of(app, "#detail-text")
        assert "[bold red]x[/]" in detail and "\x1b" not in detail and "\x07" not in detail
        table = app.query_one("#table", DataTable)
        cells = [str(table.get_cell_at(Coordinate(0, c))) for c in range(8)]
        assert "[bold red]x[/]" in cells[5] and all("\x1b" not in c for c in cells)
        # a hostile CLI error is escaped on the status line too
        backend.answers["ack"] = CliResult(1, {"error": {"code": "internal", "message": HOSTILE}})
        await pilot.press("k")
        await settle(app, pilot)
        status = text_of(app, "#status")
        assert "\x1b" not in status and "\x00" not in status and "[bold red]x[/]" in status

    run_app(make(backend), script)


def test_a_huge_name_does_not_break_the_layout() -> None:
    backend = FakeBackend([agent(IDS[0], name="n" * 200000, labels=["l" * 200000] * 3)])

    async def script(app: BrowseApp, pilot: Any) -> None:
        await pilot.press("enter")
        await settle(app, pilot)
        assert len(text_of(app, "#detail-text")) < 5000

    run_app(make(backend), script)


def test_json_roundtrip_of_intent_matches_schema() -> None:
    from agent_spawn.tui.model import intent_text

    schema_registry.validate(
        json.loads(intent_text("resume", IDS)), "agent-spawn/cli/browse-result.json"
    )
