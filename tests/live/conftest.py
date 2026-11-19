"""Fixtures and gating for ``tests/live``.

Every test collected from this directory is marked ``live`` here (a test cannot forget the
marker and leak into CI) and must carry ``@pytest.mark.live_scenario("<name>")`` naming a
scenario of ``scenarios.py``. Tests are skipped unless ``AS_LIVE=1`` (``make live`` sets it).
``LIVE_SCENARIOS`` (names, ``wave:<name>``, ``billed``, ``unbilled``) selects scenarios; ``-k``
works as usual. See docs/live-testing.md.
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import atexit
import json
import shutil
import signal
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests import isolation

from . import harness
from .harness import LiveError, LiveTimeout, World
from .scenarios import SCENARIOS, Scenario, parse_selection

LIVE_DIR = Path(__file__).resolve().parent
#: Read at import, before the autouse fixtures replace ``HOME`` and drop ``AGENT_SPAWN_*``.
ENVIRON: dict[str, str] = dict(isolation.ORIGINAL_ENV)
REAL_HOME = Path(ENVIRON.get("HOME", "/nonexistent"))

_state: dict[str, Any] = {
    "observations": [],
    "started": None,
    "login": None,
    "budget": None,
    "restore_signals": None,
}


def _cleanup_private_files() -> None:
    """Delete the login copy and the suite directory (idempotent; also run at exit, so a
    pytest that dies without ``pytest_sessionfinish`` leaves no credential copy behind)."""
    login = _state.get("login")
    if login is not None:
        login.remove()
    suite_dir = _state.get("suite_dir")
    if suite_dir is not None:
        shutil.rmtree(suite_dir, ignore_errors=True)


atexit.register(_cleanup_private_files)


def pytest_sessionstart(session: pytest.Session) -> None:
    if harness.opted_in(ENVIRON) and _state["restore_signals"] is None:
        _state["restore_signals"] = harness.install_termination_handlers()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live_scenario(name): the scenario of tests/live/scenarios.py a live test belongs to",
    )


def _scenario_of(item: pytest.Item) -> Scenario | None:
    marker = item.get_closest_marker("live_scenario")
    if marker is None:
        return None
    return SCENARIOS.get(marker.args[0])


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    try:
        selected = parse_selection(ENVIRON.get(harness.SCENARIOS_ENV))
    except ValueError as exc:
        raise pytest.UsageError(f"{harness.SCENARIOS_ENV}: {exc}") from None
    opted = harness.opted_in(ENVIRON)
    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        if not Path(str(item.path)).is_relative_to(LIVE_DIR):
            kept.append(item)
            continue
        item.add_marker(pytest.mark.live)
        marker = item.get_closest_marker("live_scenario")
        if marker is None or marker.args[0] not in SCENARIOS:
            name = marker.args[0] if marker else None
            raise pytest.UsageError(
                f"{item.nodeid}: a live test must name a scenario of tests/live/scenarios.py "
                f"with @pytest.mark.live_scenario (got {name!r})"
            )
        if selected is not None and marker.args[0] not in selected:
            deselected.append(item)
            continue
        if ENVIRON.get("CI"):
            pass  # tests/conftest.py skips every live test in CI with its own reason
        elif not opted:
            item.add_marker(
                pytest.mark.skip(reason=f"live tests need {harness.OPT_IN_ENV}=1 (make live)")
            )
        else:
            scenario = SCENARIOS[marker.args[0]]
            if scenario.skip:
                item.add_marker(pytest.mark.skip(reason=f"{scenario.name}: {scenario.skip}"))
        kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


# --- suite state -----------------------------------------------------------------------------


def _budget() -> harness.SessionBudget:
    if _state["budget"] is None:
        directory = Path(tempfile.mkdtemp(prefix="as-live-suite-", dir="/tmp"))
        directory.chmod(0o700)
        _state["suite_dir"] = directory
        _state["budget"] = harness.SessionBudget(
            directory / "claude-sessions", harness.max_sessions(ENVIRON)
        )
    return _state["budget"]


def _login(auth: harness.Auth) -> harness.LoginCache | None:
    if auth.source is None:
        return None
    if _state["login"] is None:
        _state["login"] = harness.new_login_cache(auth.source)
    return _state["login"]


@pytest.fixture(scope="session")
def live_tools() -> dict[str, harness.Tool | None]:
    return harness.discover_tools(ENVIRON)


def _alarm_handler(signum: int, frame: Any) -> None:
    raise LiveTimeout("the test's wall-clock bound ran out")


@pytest.fixture(autouse=True)
def live_clock(request: pytest.FixtureRequest) -> Iterator[float | None]:
    """The per-test and per-suite wall-clock bounds (SIGALRM raises ``LiveTimeout``)."""
    if request.node.get_closest_marker("live") is None or not harness.opted_in(ENVIRON):
        yield None
        return
    scenario = _scenario_of(request.node)
    if _state["started"] is None:
        _state["started"] = time.monotonic()
    suite_left = harness.env_int(
        ENVIRON, harness.SUITE_TIMEOUT_ENV, harness.DEFAULT_SUITE_TIMEOUT
    ) - (time.monotonic() - _state["started"])
    if suite_left <= 0:
        pytest.exit(f"{harness.SUITE_TIMEOUT_ENV} ran out: the live suite stopped", returncode=3)
    default = 420 if scenario is not None and scenario.billed else 180
    per_test = harness.env_int(
        ENVIRON,
        harness.TEST_TIMEOUT_ENV,
        scenario.timeout if scenario and scenario.timeout else default,
    )
    bound = min(per_test, suite_left)
    previous = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, bound)
    try:
        yield time.monotonic() + bound
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _require_tools(scenario: Scenario, tools: dict[str, harness.Tool | None]) -> None:
    for name in scenario.tools:
        if tools.get(name) is not None:
            continue
        message = (
            f"{scenario.name}: no real {name} (set {harness.TOOL_ENV[name]} or put it on PATH)"
        )
        if name in scenario.optional or not harness.strict(ENVIRON):
            pytest.skip(message)
        raise LiveError(message)
    if scenario.min_claude_session:
        tool = tools.get("claude-session")
        if tool is not None and tool.at_least(scenario.min_claude_session) is False:
            pytest.skip(
                f"{scenario.name}: needs claude-session >= {scenario.min_claude_session}, "
                f"found {tool.version_text}"
            )


def _make_world(
    request: pytest.FixtureRequest,
    tools: dict[str, harness.Tool | None],
    deadline: float | None,
    *,
    billed: bool,
) -> Iterator[World]:
    scenario = _scenario_of(request.node)
    if scenario is None:
        raise LiveError("a live test without a registered scenario")
    if billed != scenario.billed:
        raise LiveError(f"{scenario.name}: uses the {'billed' if billed else 'unbilled'} fixture")
    _require_tools(scenario, tools)
    budget = _budget()
    auth = None
    login = None
    if billed:
        auth = harness.resolve_auth(ENVIRON, REAL_HOME)
        if auth is None:
            message = (
                f"{scenario.name}: no credentials: set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN, "
                f"or {harness.COPY_LOGIN_ENV}=1 to copy the login file into the private home"
            )
            if harness.strict(ENVIRON):
                raise LiveError(message)
            pytest.skip(message)
        budget.check_start(scenario.name, scenario.sessions_per_test)
        login = _login(auth)
    root = harness.new_root()
    world = World(
        root,
        tools,
        ENVIRON,
        budget=budget,
        auth=auth,
        login=login,
        real_home=REAL_HOME,
        deadline=deadline,
    )
    used_before = budget.used
    refused_before = len(harness.refusals(budget.counter))
    per_test = scenario.sessions_per_test
    try:
        if billed:
            world.install_claude()
            world.install_auth()
        yield world
    finally:
        _state["observations"].extend(world.observations)
        world.close()
    if billed:
        problems = budget.check_end(scenario.name, per_test, used_before, refused_before)
        if problems:
            raise LiveError("; ".join(problems))


@pytest.fixture
def world(
    request: pytest.FixtureRequest, live_tools: dict[str, Any], live_clock: float | None
) -> Iterator[World]:
    """A private world for an unbilled scenario: no Claude Code, no credentials."""
    yield from _make_world(request, live_tools, live_clock, billed=False)


@pytest.fixture
def billed_world(
    request: pytest.FixtureRequest, live_tools: dict[str, Any], live_clock: float | None
) -> Iterator[World]:
    """A private world for a billed scenario: the counting ``claude`` wrapper first on PATH and
    credentials installed in the private home."""
    yield from _make_world(request, live_tools, live_clock, billed=True)


def pytest_sessionfinish(session: pytest.Session) -> None:
    restore = _state.get("restore_signals")
    if restore is not None:
        restore()
        _state["restore_signals"] = None
    path = ENVIRON.get(harness.OBSERVATIONS_ENV)
    if path and _state["observations"]:
        with open(path, "a", encoding="utf-8") as handle:
            for key, value in _state["observations"]:
                handle.write(json.dumps({"key": key, "value": value}, default=str) + "\n")
    budget = _state.get("budget")
    if budget is not None:
        _state["summary"] = f"{budget.used} claude sessions counted (cap {budget.maximum})"
    _cleanup_private_files()


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if _state.get("summary"):
        terminalreporter.write_line(f"live: {_state['summary']}")
    for key, value in _state["observations"]:
        terminalreporter.write_line(f"live observation: {key} = {value!r}")
