"""Shared runtime of the fake binaries: argv parsing, invocation log, per-test scripts.

The fakes are standard-library only (they run under whatever ``python3`` is on ``PATH``), log
every invocation, parse argv with the real tool's grammar (an argv the real tool would refuse
is refused here too, with the same exit code), and answer from canned outputs that the tests
validate against the pinned sibling schemas. See ``tests/fakebin/FIDELITY.md``.

Environment (set by the ``fakebin`` fixture; a fake refuses to run without ``FAKEBIN_LOG``):

* ``FAKEBIN_LOG``: JSONL file; one ``invoke`` record per call, one ``exit`` record when it ends.
* ``FAKEBIN_SCRIPT``: optional JSON file ``{"rules": [...]}`` of scripted responses.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import signal
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

LOGGED_ENV_PREFIXES = ("CSESSION_", "TERMCTL_", "WTCTL_", "AGENT_SPAWN_")
LOGGED_ENV_NAMES = ("HOME", "XDG_STATE_HOME", "XDG_CONFIG_HOME", "TMUX", "TMUX_TMPDIR", "PATH")
STDIN_LOG_LIMIT = 65536
FIXED_TS = "2026-11-03T10:00:00Z"

BOOL, VALUE, MULTI = "bool", "value", "multi"


class Exit(Exception):
    """End the fake with an exit code, after printing ``stdout`` and ``stderr``."""

    def __init__(self, code: int, stdout: str = "", stderr: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.stdout = stdout
        self.stderr = stderr


class Usage(Exception):
    """An argv the real tool refuses with exit 2."""


@dataclass
class Parsed:
    flags: dict[str, Any] = field(default_factory=dict)
    positionals: list[str] = field(default_factory=list)
    positional_indexes: list[int] = field(default_factory=list)
    after_dd: list[str] | None = None
    tokens: list[str] = field(default_factory=list)

    def has(self, name: str) -> bool:
        return name in self.flags

    def get(self, name: str, default: Any = None) -> Any:
        return self.flags.get(name, default)

    def multi(self, name: str) -> list[str]:
        return list(self.flags.get(name, []))


def parse_args(
    argv: list[str],
    flags: dict[str, str],
    *,
    dd_is_argv: bool = False,
    stop_at_dd: bool = True,
) -> Parsed:
    """Parse ``argv`` with getopt-like rules shared by the three tools.

    A value flag always consumes the next token (even one starting with ``-``); ``--flag=value``
    is equivalent; a boolean flag rejects ``=value``; ``--`` ends the options, and what follows
    is ``after_dd`` (``dd_is_argv``) or more positionals; a lone ``-`` is a positional; an
    unknown option, a missing value or a combined short option is a usage error.
    """
    parsed = Parsed(tokens=list(argv))
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--" and stop_at_dd:
            rest = argv[i + 1 :]
            if dd_is_argv:
                parsed.after_dd = rest
            else:
                for offset, item in enumerate(rest):
                    parsed.positionals.append(item)
                    parsed.positional_indexes.append(i + 1 + offset)
            return parsed
        if token.startswith("--") and token != "--":
            name, eq, value = token.partition("=")
            kind = flags.get(name)
            if kind is None:
                raise Usage(f"unrecognized argument: {token}")
            if kind == BOOL:
                if eq:
                    raise Usage(f"{name} takes no value")
                parsed.flags[name] = True
            else:
                if not eq:
                    i += 1
                    if i >= len(argv):
                        raise Usage(f"{name} requires a value")
                    value = argv[i]
                if kind == MULTI:
                    parsed.flags.setdefault(name, []).append(value)
                else:
                    parsed.flags[name] = value
        elif token.startswith("-") and token != "-":
            raise Usage(f"unrecognized argument: {token}")
        else:
            parsed.positionals.append(token)
            parsed.positional_indexes.append(i)
        i += 1
    return parsed


def require_pattern(value: str, pattern: str, what: str) -> None:
    if re.fullmatch(pattern, value) is None:
        raise Usage(f"invalid {what}: {value!r}")


def base32_id(prefix: str, number: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz234567"
    chars = []
    for _ in range(8):
        chars.append(alphabet[number % 32])
        number //= 32
    return prefix + "".join(reversed(chars))


def deep_merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        out = dict(base)  # pyright: ignore[reportUnknownArgumentType]
        for key, value in overlay.items():  # pyright: ignore[reportUnknownVariableType]
            out[key] = deep_merge(out.get(key), value)
        return out
    return overlay


# --- invocation log and scripts -------------------------------------------------------------


def _log_path() -> Path:
    raw = os.environ.get("FAKEBIN_LOG")
    if not raw:
        sys.stderr.write("fakebin: FAKEBIN_LOG is not set; refusing to run outside the fixture\n")
        raise SystemExit(70)
    return Path(raw)


def _append_line(path: Path, record: dict[str, Any]) -> None:
    line = (json.dumps(record, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    fd = os.open(str(path) + ".lock", os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def counter(name: str) -> int:
    """A persistent, process-safe counter next to the log; returns the new value (from 1)."""
    path = Path(str(_log_path()) + ".counts.json")
    with locked(path):
        data: dict[str, int] = {}
        if path.exists():
            data = json.loads(path.read_text())
        data[name] = data.get(name, 0) + 1
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)
        return data[name]


def find_rule(tool: str, command: str, argv: list[str]) -> dict[str, Any] | None:
    """The first scripted rule that matches and still has uses left."""
    script = os.environ.get("FAKEBIN_SCRIPT")
    if not script or not Path(script).exists():
        return None
    rules = json.loads(Path(script).read_text()).get("rules", [])
    for index, rule in enumerate(rules):
        if rule.get("tool") not in (None, tool):
            continue
        if rule.get("command") not in (None, command):
            continue
        if any(token not in argv for token in rule.get("argv_contains", [])):
            continue
        times = rule.get("times")
        if times is not None and counter(f"rule-{index}") > times:
            continue
        return rule
    return None


@dataclass
class Context:
    tool: str
    argv: list[str]
    invoke_id: str
    json_mode: bool = False
    stdin_text: str | None = None
    signaled: list[int] = field(default_factory=list)
    _rule: dict[str, Any] | None = None
    _rule_loaded: bool = False
    _behaved: bool = False

    def rule(self, command: str) -> dict[str, Any] | None:
        """The scripted rule for this call (looked up, and counted, once)."""
        if not self._rule_loaded:
            self._rule = find_rule(self.tool, command, self.argv)
            self._rule_loaded = True
        return self._rule

    def response(self, command: str) -> dict[str, Any]:
        rule = self.rule(command)
        return {} if rule is None else dict(rule.get("respond", {}))

    def check_scripted_error(self, command: str) -> None:
        """End now with a scripted error, before the caller changes any state.

        A real tool that refuses (``locked``, ``busy``, a veto) changed nothing, so handlers with
        side effects call this first.
        """
        response = self.response(command)
        if "error" not in response:
            return
        self.behave(command)
        error = dict(response["error"])
        body = json.dumps({"error": error}) + "\n" if self.json_mode else ""
        raise Exit(
            int(response.get("exit", 1)),
            body,
            f"{self.tool}: {error.get('message', '')}\n{response.get('stderr', '')}",
        )

    def behave(self, command: str) -> None:
        """Apply the scripted sleep / hang / signal behavior now (once)."""
        if self._behaved:
            return
        self._behaved = True
        response = self.response(command)
        install_signal_behavior(response.get("on_signal", {}), self.signaled)
        if response.get("sleep"):
            time.sleep(float(response["sleep"]))
        if response.get("hang"):
            # Block until a signal ends the process (default action) or on_signal handles it.
            while True:
                time.sleep(0.05)

    def stdin_is_tty(self) -> bool:
        return sys.stdin is not None and sys.stdin.isatty()

    def read_stdin(self) -> str:
        """Read stdin to EOF (only commands that really read it call this; it is logged)."""
        text = self.stdin_text
        if text is None:
            text = sys.stdin.buffer.read().decode("utf-8", "replace")
            self.stdin_text = text
        return text

    def fail(self, code: int, error_code: str, message: str, **extra: Any) -> NoReturn:
        error: dict[str, Any] = {"code": error_code, "message": message, **extra}
        stdout = json.dumps({"error": error}) + "\n" if self.json_mode else ""
        raise Exit(code, stdout, f"{self.tool}: {message}\n")

    def usage(self, message: str) -> NoReturn:
        self.fail(2, "usage", message)


Handler = Callable[[Context, list[str]], tuple[int, Any]]


def run_fake(
    tool: str, argv: list[str], dispatch: Callable[[Context], tuple[int, Any, str]]
) -> int:
    """Log the call, run ``dispatch`` and apply scripted behavior; returns the exit code.

    ``dispatch(ctx)`` returns ``(exit_code, canned_output, command)`` where ``command`` is the
    canonical subcommand (``"label add"``) a script rule matches against, or raises ``Exit``.
    The script's response may replace or merge into the canned output, change the exit code,
    sleep, hang until signaled, or choose how a signal ends the fake.
    """
    log = _log_path()
    invoke_id = f"{time.time_ns()}-{os.getpid()}"
    ctx = Context(tool, argv, invoke_id)
    env = {
        name: value
        for name, value in os.environ.items()
        if name.startswith(LOGGED_ENV_PREFIXES) or name in LOGGED_ENV_NAMES
    }
    _append_line(
        log,
        {
            "event": "invoke",
            "id": invoke_id,
            "tool": tool,
            "argv": argv,
            "cwd": os.getcwd(),
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "env": env,
            "stdin_is_tty": sys.stdin is not None and sys.stdin.isatty(),
        },
    )
    code = 1
    signaled = ctx.signaled

    def finish() -> None:
        _append_line(
            log,
            {
                "event": "exit",
                "id": invoke_id,
                "tool": tool,
                "exit_code": code,
                "signals": signaled,
                "stdin": None if ctx.stdin_text is None else ctx.stdin_text[:STDIN_LOG_LIMIT],
            },
        )

    try:
        try:
            code, output, command = dispatch(ctx)
            output, code = apply_rule(ctx, command, output, code)
            emit(ctx, output)
        except Exit as stop:
            code = stop.code
            sys.stdout.write(stop.stdout)
            sys.stderr.write(stop.stderr)
        except Usage as problem:
            ctx.json_mode = any(t == "--json" or t.startswith("--json=") for t in _before_dd(argv))
            code = 2
            if ctx.json_mode:
                body = {"error": {"code": "usage", "message": str(problem)}}
                sys.stdout.write(json.dumps(body) + "\n")
            sys.stderr.write(f"{tool}: {problem}\n")
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        finish()
    return code


def _before_dd(argv: list[str]) -> list[str]:
    return argv[: argv.index("--")] if "--" in argv else argv


def emit(ctx: Context, output: Any) -> None:
    if output is None:
        return
    if isinstance(output, str):
        sys.stdout.write(output)
    elif ctx.json_mode:
        sys.stdout.write(json.dumps(output) + "\n")


def apply_rule(ctx: Context, command: str, output: Any, code: int) -> tuple[Any, int]:
    response = ctx.response(command)
    ctx.behave(command)
    if "error" in response:
        error = dict(response["error"])
        sys.stderr.write(f"{ctx.tool}: {error.get('message', '')}\n")
        sys.stderr.write(response.get("stderr", ""))
        return {"error": error}, int(response.get("exit", code or 1))
    if "stdout" in response:
        output = response["stdout"]
    if "merge" in response:
        output = deep_merge(output, response["merge"])
    if "exit" in response:
        code = int(response["exit"])
    sys.stderr.write(response.get("stderr", ""))
    return output, code


def install_signal_behavior(spec: dict[str, Any], signaled: list[int]) -> None:
    """``{"SIGTERM": {"exit": 143}}`` exits with that code; ``"ignore"`` ignores the signal."""
    for name, action in spec.items():
        number = getattr(signal, name)
        if action == "ignore":
            signal.signal(number, signal.SIG_IGN)
            continue

        def handler(signum: int, _frame: Any, action: dict[str, Any] = action) -> None:
            signaled.append(signum)
            raise Exit(int(action.get("exit", 128 + signum)), "", action.get("stderr", ""))

        signal.signal(number, handler)
