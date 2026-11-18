"""The browse screen (Textual). Importing this module needs the ``tui`` extra.

A list of agents with filters, a detail pane and the actions of interface.md "TUI". Nothing here
touches the state database: rows come from ``list``, details from ``show`` and every action is the
CLI command of the same name, run through a ``Backend``. Text from agents only reaches a widget
through ``model.shown`` / ``model.agent_cells`` (escaped, bounded, plain), and the widgets are
built without markup.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any, ClassVar

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult, SuspendNotSupported
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Input, Static

from .model import (
    Backend,
    CliResult,
    FilterState,
    agent_cells,
    detail_text,
    parse_agents,
    shown,
)

__all__ = ["BrowseApp", "ConfirmScreen", "HelpScreen"]

COLUMNS = ("", "id", "status", "attention", "kind", "name", "labels", "age")
WAIT_TIMEOUT = "60s"
HELP_TEXT = """\
up/down   move             enter  load the full detail (show)
space     mark / unmark    a      attach (one agent)
r         resume marked agents, or the highlighted one (interactive, in a window)
k         ack              s      stop (asks first)
t         teardown (asks first)        w   wait (up to 60s)
/         label filter     f      cycle the status filter
n         attention only   esc    clear the filters
R         refresh          ?      this help
q         quit
"""


class ConfirmScreen(ModalScreen[bool]):
    """The TUI's own confirmation dialog; the default answer is No."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "answer(True)", "Yes"),
        Binding("n,escape", "answer(False)", "No"),
    ]

    def __init__(self, question: str) -> None:
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm"):
            yield Static(self._question, id="question", markup=False)
            with Horizontal():
                yield Button("Yes (y)", id="yes")
                yield Button("No (n)", id="no")

    def on_mount(self) -> None:
        self.query_one("#no", Button).focus()

    def action_answer(self, answer: bool) -> None:
        self.dismiss(answer)

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "yes")


class HelpScreen(ModalScreen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape,q,question_mark", "close", "Close")]

    def compose(self) -> ComposeResult:
        yield Static(HELP_TEXT, id="help", markup=False)

    def action_close(self) -> None:
        self.dismiss(None)


class BrowseApp(App[None]):
    TITLE = "agent-spawn browse"
    CSS = """
    #body { height: 1fr; }
    #list { width: 60%; }
    #detail { width: 40%; border-left: solid $primary; padding: 0 1; }
    #empty { padding: 1 2; }
    #filter { display: none; }
    #filter.open { display: block; }
    #status { height: 1; padding: 0 1; }
    #filters { height: 1; padding: 0 1; }
    ConfirmScreen, HelpScreen { align: center middle; }
    #confirm, #help { width: 60; height: auto; border: thick $primary; padding: 1 2; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit_browse", "Quit"),
        Binding("question_mark", "help", "Help"),
        Binding("slash", "filter", "Filter"),
        Binding("enter", "open", "Open"),
        Binding("a", "attach", "Attach"),
        Binding("r", "resume", "Resume"),
        Binding("k", "ack", "Ack"),
        Binding("s", "stop", "Stop"),
        Binding("t", "teardown", "Teardown"),
        Binding("w", "wait", "Wait"),
        Binding("R", "refresh_list", "Refresh"),
        Binding("space", "mark", "Mark"),
        Binding("f", "cycle_status", "Status"),
        Binding("n", "toggle_attention", "Attention"),
        Binding("escape", "clear_filters", "Clear", show=False),
    ]

    def __init__(
        self,
        backend: Backend,
        *,
        focus: str | None = None,
        record_intent: bool = False,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        super().__init__()
        self.backend = backend
        self.focus_id = focus
        self.record_intent = record_intent
        self.clock = clock
        self.intent: tuple[str, list[str]] | None = None
        self.filters = FilterState()
        self.agents: list[dict[str, Any]] = []
        self.visible: list[dict[str, Any]] = []
        self.marked: set[str] = set()
        self.details: dict[str, dict[str, Any]] = {}
        self.detail_errors: dict[str, str] = {}
        self.list_error: str | None = None
        self._busy: str | None = None
        self._waiting: str | None = None

    # --- layout ------------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="list"):
                yield Input(placeholder="label filter (enter keeps, esc clears)", id="filter")
                yield DataTable(id="table", cursor_type="row", zebra_stripes=False)
                yield Static("", id="empty", markup=False)
            with VerticalScroll(id="detail"):
                yield Static("", id="detail-text", markup=False)
        yield Static("", id="filters", markup=False)
        yield Static("loading...", id="status", markup=False)
        yield Footer()

    async def on_mount(self) -> None:
        self.theme = "textual-ansi"  # the terminal's own colors
        table = self.query_one("#table", DataTable)
        for label in COLUMNS:
            table.add_column(label)
        table.focus()
        self.run_worker(self._reload(initial=True), group="reload")

    def on_unmount(self) -> None:
        self.backend.close()

    # --- status and rendering ----------------------------------------------------------------

    def status(self, text: str) -> None:
        self.query_one("#status", Static).update(shown(text, 300))

    def _current(self) -> dict[str, Any] | None:
        table = self.query_one("#table", DataTable)
        if not self.visible or not 0 <= table.cursor_row < len(self.visible):
            return None
        return self.visible[table.cursor_row]

    def _render_table(self, keep: str | None = None) -> None:
        table = self.query_one("#table", DataTable)
        empty = self.query_one("#empty", Static)
        self.visible = [a for a in self.agents if self.filters.matches(a)]
        now = self.clock()
        table.clear()
        for agent in self.visible:
            cells = agent_cells(agent, now, marked=agent["agent_id"] in self.marked)
            table.add_row(*(Text(cell) for cell in cells), key=agent["agent_id"])
        if self.list_error is not None:
            message = f"Could not list agents: {self.list_error}"
        elif not self.agents:
            message = "No agents yet. Start one with `agent-spawn launch`."
        elif not self.visible:
            message = "No agent matches the filters (esc clears them)."
        else:
            message = ""
        empty.update(message)
        empty.display = bool(message)
        table.display = not message
        if keep is not None:
            for index, agent in enumerate(self.visible):
                if agent["agent_id"] == keep:
                    table.move_cursor(row=index)
                    break
        self.query_one("#filters", Static).update(
            f"filters: {self.filters.summary()}  ({len(self.visible)}/{len(self.agents)} shown)"
        )
        self._render_detail()

    def _render_detail(self) -> None:
        agent = self._current()
        pane = self.query_one("#detail-text", Static)
        if agent is None:
            pane.update("no agent selected")
            return
        agent_id = agent["agent_id"]
        text = detail_text(agent, self.details.get(agent_id))
        if agent_id in self.detail_errors:
            text += f"\n\nshow failed: {self.detail_errors[agent_id]}"
        pane.update(text)

    @on(DataTable.RowHighlighted)
    def _highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._render_detail()

    @on(DataTable.RowSelected)
    def _selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.action_open()

    # --- loading -----------------------------------------------------------------------------

    async def _reload(self, *, initial: bool = False) -> None:
        keep = self._current()
        keep_id = keep["agent_id"] if keep else None
        result = await asyncio.to_thread(self.backend.call, ["list"])
        if result.ok:
            self.agents = parse_agents(result.payload)
            self.list_error = None
            self.marked &= {a["agent_id"] for a in self.agents}
            self.details.clear()
            self.detail_errors.clear()
            self.status(f"{len(self.agents)} agent(s)")
        else:
            self.list_error = result.describe()
            self.status(f"list failed, {self.list_error}")
        self._render_table(keep_id)
        if initial and self.focus_id is not None:
            await self._focus_on(self.focus_id)

    async def _focus_on(self, wanted: str) -> None:
        prefix = wanted if wanted.startswith("ag-") else f"ag-{wanted}"
        matches = [a["agent_id"] for a in self.agents if a["agent_id"].startswith(prefix)]
        if len(matches) != 1:
            kind = "several agents match" if matches else "no listed agent matches"
            self.status(f"--focus {shown(wanted, 40)}: {kind}")
            return
        self._render_table(matches[0])
        await self._load_detail(matches[0])

    async def _load_detail(self, agent_id: str) -> None:
        self.status(f"show {agent_id}...")
        result = await asyncio.to_thread(self.backend.call, ["show", agent_id])
        if result.ok and isinstance(result.payload, dict):
            self.details[agent_id] = result.payload  # pyright: ignore[reportUnknownMemberType]
            self.detail_errors.pop(agent_id, None)
            self.status(f"show {agent_id}: ok")
        else:
            self.detail_errors[agent_id] = result.describe() or "unusable answer"
            self.status(f"show {agent_id} failed, {self.detail_errors[agent_id]}")
        self._render_detail()

    # --- actions: navigation and filters -----------------------------------------------------

    def action_open(self) -> None:
        agent = self._current()
        if agent is None:
            self.status("no agent selected")
            return
        self.run_worker(self._load_detail(agent["agent_id"]), group="detail", exclusive=True)

    def action_refresh_list(self) -> None:
        if self._busy is not None:
            self.status(f"busy: {self._busy}")
            return
        self.status("refreshing...")
        self.run_worker(self._reload(), group="reload", exclusive=True)

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.add_class("open")
        box.focus()

    @on(Input.Changed, "#filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        event.stop()
        current = self._current()
        self.filters.label = event.value
        self._render_table(current["agent_id"] if current else None)

    @on(Input.Submitted, "#filter")
    def _filter_done(self, event: Input.Submitted) -> None:
        event.stop()
        self.query_one("#table", DataTable).focus()

    def on_key(self, event: Any) -> None:
        # escape inside the filter box clears it (App bindings do not see it first)
        if event.key == "escape" and self.focused is self.query_one("#filter", Input):
            event.stop()
            self.action_clear_filters()

    def action_clear_filters(self) -> None:
        current = self._current()
        self.filters = FilterState()
        box = self.query_one("#filter", Input)
        box.value = ""
        box.remove_class("open")
        self.query_one("#table", DataTable).focus()
        self._render_table(current["agent_id"] if current else None)

    def action_cycle_status(self) -> None:
        current = self._current()
        self.filters.status = self.filters.next_status()
        self._render_table(current["agent_id"] if current else None)

    def action_toggle_attention(self) -> None:
        current = self._current()
        self.filters.attention_only = not self.filters.attention_only
        self._render_table(current["agent_id"] if current else None)

    def action_mark(self) -> None:
        agent = self._current()
        if agent is None:
            return
        agent_id = agent["agent_id"]
        self.marked ^= {agent_id}
        self._render_table(agent_id)

    # --- actions: quit and intents -----------------------------------------------------------

    def action_quit_browse(self) -> None:
        if self.record_intent:
            self.intent = ("back", [])
        self.exit()

    def _targets(self) -> list[str]:
        """Marked agents in list order, else the highlighted one."""
        marked = [a["agent_id"] for a in self.agents if a["agent_id"] in self.marked]
        if marked:
            return marked
        current = self._current()
        return [current["agent_id"]] if current else []

    def action_attach(self) -> None:
        targets = self._targets()
        if not targets:
            self.status("no agent selected")
            return
        if len(targets) != 1:
            self.status("attach takes one agent: unmark the others (space)")
            return
        if self.record_intent:
            self.intent = ("attach", targets)
            self.exit()
            return
        self._start(f"attach {targets[0]}", self._attach(targets[0]))

    def action_resume(self) -> None:
        targets = self._targets()
        if not targets:
            self.status("no agent selected")
            return
        if self.record_intent:
            self.intent = ("resume", targets)
            self.exit()
            return
        self._start(f"resume {' '.join(targets)}", self._resume(targets))

    # --- actions that run the CLI ------------------------------------------------------------

    def _start(self, label: str, work: Coroutine[Any, Any, None]) -> None:
        if self._busy is not None:
            work.close()
            self.status(f"busy: {self._busy}")
            return
        self._busy = label
        self.status(f"{label}...")
        self.run_worker(self._guarded(work), group="action")

    async def _guarded(self, work: Coroutine[Any, Any, None]) -> None:
        try:
            await work
        finally:
            self._busy = None
        await self._reload()

    async def _call(self, command: list[str], *, timeout: float | None = None) -> CliResult:
        return await asyncio.to_thread(self.backend.call, command, acting=True, timeout=timeout)

    async def _attach(self, agent_id: str) -> None:
        result = await self._call(["attach", agent_id])
        payload = result.payload if isinstance(result.payload, dict) else {}
        if not result.ok:
            self.status(f"attach {agent_id} failed, {result.describe()}")
            return
        argv = payload.get("argv")  # pyright: ignore[reportUnknownMemberType]
        if payload.get("action") == "attach" and isinstance(argv, list) and argv:  # pyright: ignore[reportUnknownMemberType]
            code = await self._on_terminal([str(part) for part in argv])  # pyright: ignore[reportUnknownVariableType]
            self.status(f"attach {agent_id}: back (exit {code})")
        else:
            self.status(f"attach {agent_id}: switched")

    async def _on_terminal(self, argv: list[str]) -> int:
        """Give the terminal to ``argv`` (tmux attach) and take it back afterwards."""
        try:
            with self.suspend():
                return self.backend.run_on_terminal(argv)
        except SuspendNotSupported:
            self.status("this terminal cannot be suspended; attach from a shell instead")
            return 1

    async def _resume(self, targets: list[str]) -> None:
        notes: list[str] = []
        for agent_id in targets:
            result = await self._call(
                ["resume", agent_id, "--mode", "interactive", "--placement", "window"]
            )
            notes.append(f"{agent_id} ok" if result.ok else f"{agent_id} {result.describe()}")
        self.status("resume: " + "; ".join(notes))

    def _simple(self, verb: str, command: list[str], agent_id: str) -> None:
        async def work() -> None:
            result = await self._call(command)
            self.status(f"{verb} {agent_id}: " + ("ok" if result.ok else result.describe()))

        self._start(f"{verb} {agent_id}", work())

    def action_ack(self) -> None:
        agent = self._current()
        if agent is None:
            self.status("no agent selected")
            return
        self._simple("ack", ["ack", agent["agent_id"]], agent["agent_id"])

    def _confirm(self, verb: str, command_of: Callable[[str], list[str]], warning: str) -> None:
        agent = self._current()
        if agent is None:
            self.status("no agent selected")
            return
        agent_id = agent["agent_id"]
        question = f"{verb} {agent_id} ({shown(agent.get('name'), 60)})?\n{warning}"

        def answered(yes: bool | None) -> None:
            if yes:
                self._simple(verb, command_of(agent_id), agent_id)
            else:
                self.status(f"{verb} cancelled")

        self.push_screen(ConfirmScreen(question), answered)

    def action_stop(self) -> None:
        self._confirm("stop", lambda i: ["stop", i, "--yes"], "The agent's current run ends.")

    def action_teardown(self) -> None:
        self._confirm(
            "teardown",
            lambda i: ["teardown", i, "--yes"],
            "Its worktree is removed when it owns one.",
        )

    def action_wait(self) -> None:
        agent = self._current()
        if agent is None:
            self.status("no agent selected")
            return
        agent_id = agent["agent_id"]
        if self._waiting is not None:
            self.status(f"already waiting for {self._waiting}")
            return
        self._waiting = agent_id
        self.status(f"waiting for {agent_id} (up to {WAIT_TIMEOUT})...")
        self.run_worker(self._wait(agent_id), group="wait")

    async def _wait(self, agent_id: str) -> None:
        try:
            result = await self._call(["wait", agent_id, "--timeout", WAIT_TIMEOUT], timeout=90)
        finally:
            self._waiting = None
        payload = result.payload if isinstance(result.payload, dict) else {}
        if "error" in payload or "phase" not in payload:  # pyright: ignore[reportUnknownMemberType]
            self.status(f"wait {agent_id}: {result.describe() or 'no answer'}")
        elif payload.get("timed_out"):  # pyright: ignore[reportUnknownMemberType]
            self.status(f"wait {agent_id}: still running after {WAIT_TIMEOUT}")
        else:
            self.status(
                f"wait {agent_id}: {shown(payload.get('phase'), 40)}"  # pyright: ignore[reportUnknownMemberType]
                f" outcome={shown(payload.get('outcome') or '-', 40)} (exit {result.exit_code})"  # pyright: ignore[reportUnknownMemberType]
            )
        await self._reload()
