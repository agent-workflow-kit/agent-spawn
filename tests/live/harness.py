"""Support code for the ``live`` suite: real Claude Code, tmux, claude-session, termctl, wtctl.

Opt-in. ``make live`` sets ``AS_LIVE=1``; without it every test here skips, and ``addopts``
deselects the ``live`` marker from ``make verify`` and CI. Nothing in this module starts a real
tool at import; the pure helpers (environment building, credentials, the session counter, the
scenario selection, the fake crontab, version checks) are unit-tested in
``tests/test_live_harness.py`` with no real tool.

Isolation. Every test gets a ``World``: a private root directly under ``/tmp`` (short, mode 0700,
so tmux socket paths fit) with its own ``HOME``, ``XDG_*``, ``TMPDIR``, ``TMUX_TMPDIR`` (an
existing directory, so no tool can fall back to the real default socket), agent-spawn state dir,
a ``bin`` directory that is the whole ``PATH`` of every child (symlinks to the discovered tools,
a counting ``claude`` wrapper, a tripwire ``crontab``), and per-test git repositories. Child
environments are built from an allowlist, never from the parent's environment. Nothing of the
user's real tmux server, home, crontab, wtctl or termctl state is read or written; the real home
is read only to find the login file ``AS_LIVE_COPY_LOGIN=1`` asks for (as claude-session's live
suite does).

Credentials follow claude-session's live suite. In order: ``ANTHROPIC_API_KEY``;
``CLAUDE_CODE_OAUTH_TOKEN`` (from ``claude setup-token``); the user's login file, only with
``AS_LIVE_COPY_LOGIN=1``, copied once per pytest process into a mode-0700 private directory
(``LoginCache``), installed mode 0600 into each billed test's private home and deleted at the end
of the run. A variable's value reaches only the children that need it (billed tests), is never
written to a file by this module and never printed (``Auth.__repr__`` hides it).

Cost. Every real ``claude`` started in a test is counted by the wrapper in ``World.bin``: the
wrapper appends one line to the suite's counter file, refuses (exit 97) once the count would pass
``MAX_LIVE_SESSIONS`` and refuses a launch whose ``--model`` is not a haiku model (exit 98).
``-p`` runs get ``--max-budget-usd``. Tests and the whole suite have wall-clock limits.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OPT_IN_ENV = "AS_LIVE"
STRICT_ENV = "AS_LIVE_STRICT"
COPY_LOGIN_ENV = "AS_LIVE_COPY_LOGIN"
MAX_SESSIONS_ENV = "MAX_LIVE_SESSIONS"
SCENARIOS_ENV = "LIVE_SCENARIOS"
TEST_TIMEOUT_ENV = "LIVE_TEST_TIMEOUT"
SUITE_TIMEOUT_ENV = "LIVE_SUITE_TIMEOUT"
OBSERVATIONS_ENV = "LIVE_OBSERVATIONS"
RUN_USD_ENV = "LIVE_RUN_USD"
MODEL = "haiku"

DEFAULT_MAX_SESSIONS = 25
DEFAULT_TEST_TIMEOUT = 300
DEFAULT_SUITE_TIMEOUT = 5400
DEFAULT_RUN_USD = "0.25"

TOOL_ENV = {
    "claude": "LIVE_CLAUDE",
    "claude-session": "LIVE_CLAUDE_SESSION",
    "termctl": "LIVE_TERMCTL",
    "wtctl": "LIVE_WTCTL",
    "tmux": "LIVE_TMUX",
    "git": "LIVE_GIT",
    "jobsctl": "LIVE_JOBSCTL",
}

#: Versions the design and the sibling probes were verified with. A tool older than this fails
#: ``tools-versions``; a tool whose version cannot be read (an editable install reports 0.0.0)
#: is recorded as unknown, never failed.
MINIMUMS = {
    "claude": "2.1.312",
    "claude-session": "0.38.4",
    "termctl": "0.25.0",
    "wtctl": "0.32.0",
    "tmux": "3.2",
    "git": "2.30",
}

# Parent variables copied into the private environment: locale, terminal type, proxy and
# certificate settings. Everything else (HOME, XDG_*, TMUX, SSH_AUTH_SOCK, GH_*, AWS_*, CLAUDE_*,
# ANTHROPIC_*, ...) is dropped.
PASS_THROUGH = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
)
AUTH_VARS = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")
#: Names a test's ``extra`` environment may not set: everything ``build_env`` sets.
FORBIDDEN_EXTRA = frozenset(
    {
        "HOME",
        "TMPDIR",
        "TMUX",
        "TMUX_PANE",
        "TMUX_TMPDIR",
        "PATH",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "CLAUDE_CONFIG_DIR",
        "SSH_AUTH_SOCK",
        *AUTH_VARS,
    }
)

STANDARD_PATH = ("/usr/local/bin", "/usr/bin", "/bin")


class LiveError(AssertionError):
    """A live check failed: the message names the assumption and where it is written."""


class LiveTimeout(BaseException):
    """The wall-clock bound of a test ran out (not an ``Exception``: nothing swallows it)."""


# --- environment variables of the suite ----------------------------------------------------


def opted_in(environ: Mapping[str, str]) -> bool:
    return environ.get(OPT_IN_ENV) == "1"


def strict(environ: Mapping[str, str]) -> bool:
    """Strict: a missing required tool fails its tests instead of skipping them."""
    return environ.get(STRICT_ENV, "1") != "0"


def env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None or raw == "":
        return default
    if not raw.isascii() or not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return int(raw)


def max_sessions(environ: Mapping[str, str]) -> int:
    return env_int(environ, MAX_SESSIONS_ENV, DEFAULT_MAX_SESSIONS)


# --- versions and tools --------------------------------------------------------------------

_VERSION = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


def parse_version(text: str) -> tuple[int, int, int] | None:
    found = _VERSION.search(text)
    if not found:
        return None
    return int(found.group(1)), int(found.group(2)), int(found.group(3) or 0)


def version_at_least(have: tuple[int, int, int] | None, need: str) -> bool | None:
    """``True``/``False``, or ``None`` when the version is unknown (absent or ``0.0.0``)."""
    if have is None or have == (0, 0, 0):
        return None
    wanted = parse_version(need)
    assert wanted is not None, need
    return have >= wanted


@dataclass(frozen=True)
class Tool:
    name: str
    path: str
    version_text: str
    version: tuple[int, int, int] | None

    def at_least(self, need: str) -> bool | None:
        return version_at_least(self.version, need)

    @property
    def label(self) -> str:
        return f"{self.name} {self.version_text or '?'} ({self.path})"


def tool_path(name: str, environ: Mapping[str, str]) -> str | None:
    """The tool to test: ``LIVE_<NAME>``, else the first on ``PATH`` (an absolute path)."""
    explicit = environ.get(TOOL_ENV[name])
    if explicit:
        return os.path.abspath(explicit) if os.access(explicit, os.X_OK) else None
    found = shutil.which(name, path=environ.get("PATH"))
    return os.path.abspath(found) if found else None


def read_version(name: str, path: str) -> str:
    flag = "-V" if name == "tmux" else "--version"
    try:
        done = subprocess.run(
            [path, flag],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=30,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/nonexistent", "LC_ALL": "C"},
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (done.stdout or done.stderr).strip().splitlines()[0] if (done.stdout or done.stderr) else ""


def discover_tools(environ: Mapping[str, str]) -> dict[str, Tool | None]:
    tools: dict[str, Tool | None] = {}
    for name in TOOL_ENV:
        path = tool_path(name, environ)
        if path is None:
            tools[name] = None
            continue
        text = read_version(name, path)
        tools[name] = Tool(name, path, text, parse_version(text))
    return tools


# --- credentials ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Auth:
    """How a billed test logs in: environment variables, or a login file copied from
    ``source`` to ``<home>/.claude/.credentials.json``. The values never appear in a repr."""

    env: Mapping[str, str] = field(repr=False)
    source: Path | None
    description: str

    @property
    def api_key(self) -> str | None:
        return self.env.get("ANTHROPIC_API_KEY")


def resolve_auth(environ: Mapping[str, str], real_home: Path) -> Auth | None:
    for name in AUTH_VARS:
        value = environ.get(name)
        if value:
            return Auth({name: value}, None, name)
    if environ.get(COPY_LOGIN_ENV) == "1":
        source = real_home / ".claude" / ".credentials.json"
        if source.is_file() and not source.is_symlink():
            return Auth({}, source, "copied login file")
    return None


CREDENTIALS = Path(".claude") / ".credentials.json"


class LoginCache:
    """The one copy of the user's login file a pytest process uses (``AS_LIVE_COPY_LOGIN=1``).

    Copied once into a private 0700 directory; each billed test gets it installed (mode 0600)
    into its private home, and a token Claude refreshed there is collected back so the next test
    starts from the newest one. Nothing is ever written to the real home; ``remove`` deletes the
    copy. The content is never read into a string, logged or printed.
    """

    def __init__(self, source: Path, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "credentials.json"
        self._write(source.read_bytes())

    def _write(self, data: bytes) -> None:
        tmp = self.directory / ".credentials.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        tmp.replace(self.path)

    def install(self, home: Path) -> None:
        target = home / CREDENTIALS
        target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(self.path.read_bytes())
        target.chmod(0o600)

    def collect(self, home: Path) -> None:
        target = home / CREDENTIALS
        if target.is_symlink() or not target.is_file():
            return
        data = target.read_bytes()
        if data and data != self.path.read_bytes():
            self._write(data)

    def remove(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


def new_login_cache(source: Path, base: Path | None = None) -> LoginCache:
    directory = Path(tempfile.mkdtemp(prefix="as-login-", dir=str(base or "/tmp")))
    directory.chmod(0o700)
    return LoginCache(source, directory)


def claude_json(trusted: Sequence[Path], *, api_key: str | None = None) -> dict[str, Any]:
    """The private ``~/.claude.json``: onboarding done, the directories trusted (an interactive
    session shows the trust dialog otherwise) and the custom API key approved (its last 20
    characters, which is what Claude Code stores)."""
    data: dict[str, Any] = {
        "hasCompletedOnboarding": True,
        "projects": {
            str(p): {"hasTrustDialogAccepted": True, "allowedTools": []} for p in trusted
        },
    }
    if api_key:
        data["customApiKeyResponses"] = {"approved": [api_key.strip()[-20:]], "rejected": []}
    return data


# --- the child environment -----------------------------------------------------------------


def forbidden_extra(name: str) -> bool:
    return name in FORBIDDEN_EXTRA or name.startswith("XDG_")


def build_env(
    environ: Mapping[str, str],
    *,
    root: Path,
    path_dirs: Sequence[Path] | None = None,
    auth: Auth | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The complete environment of a child, from the allowlist only.

    ``PATH`` is the private ``bin`` directory plus the standard system directories, never the
    parent's. ``TMUX_TMPDIR`` is the private (existing) directory; ``TMUX`` is never set.
    """
    home = root / "home"
    env: dict[str, str] = {k: environ[k] for k in PASS_THROUGH if k in environ}
    dirs = [str(p) for p in (path_dirs if path_dirs is not None else [root / "bin"])]
    env.update(
        {
            "PATH": os.pathsep.join([*dirs, *STANDARD_PATH]),
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(home / ".local" / "share"),
            "XDG_STATE_HOME": str(home / ".local" / "state"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "TMPDIR": str(root / "tmp"),
            "TMUX_TMPDIR": str(root / "tmuxdir"),
            "TERM": "xterm",
            "SHELL": "/bin/sh",
            "GIT_CONFIG_GLOBAL": str(home / ".gitconfig"),
            "GIT_CONFIG_NOSYSTEM": "1",
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    )
    env.setdefault("LANG", "C.UTF-8")
    if auth is not None:
        env.update(auth.env)
    for name, value in (extra or {}).items():
        if forbidden_extra(name):
            raise ValueError(f"a live test may not set {name}")
        env[name] = value
    return env


def environment_leaks(env: Mapping[str, str], real_home: Path, root: Path) -> list[str]:
    """Reasons ``env`` could reach the real user's home, tmux server or config."""
    problems: list[str] = []
    for name in ("HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
                 "TMPDIR", "TMUX_TMPDIR"):
        value = env.get(name)
        if not value or not os.path.isabs(value):
            problems.append(f"{name} is not an absolute path: {value!r}")
        elif not Path(os.path.realpath(value)).is_relative_to(Path(os.path.realpath(root))):
            problems.append(f"{name} is outside the private root: {value}")
    if Path(os.path.realpath(env.get("HOME", "/"))) == Path(os.path.realpath(real_home)):
        problems.append("HOME is the real home")
    for name in ("TMUX_PANE", "SSH_AUTH_SOCK", "XDG_RUNTIME_DIR", "CLAUDE_CONFIG_DIR"):
        if name in env:
            problems.append(f"{name} is set")
    if "TMUX" in env and not env["TMUX"].startswith(str(root) + os.sep):
        problems.append(f"TMUX names a server outside the private root: {env['TMUX']}")
    return problems


# --- the session counter and the claude wrapper -------------------------------------------

WRAPPER = """#!/bin/sh
# Counting wrapper around the real claude (written by tests/live/harness.py).
REAL={real}
COUNTER={counter}
MAX={maximum}
case "$1" in
  --version|-v|--help|-h) exec "$REAL" "$@" ;;
esac
prev=
haiku=
budget=
print=
for a in "$@"; do
  case "$prev" in
    --model) case "$a" in *haiku*) haiku=1 ;; esac ;;
  esac
  case "$a" in
    --max-budget-usd|--max-budget-usd=*) budget=1 ;;
    -p|--print) print=1 ;;
  esac
  prev="$a"
done
if [ -z "$haiku" ]; then
  echo "live claude wrapper: refusing a launch without --model haiku" >&2
  echo "no-haiku" >> "$COUNTER.refused"
  exit 98
fi
echo "$$ $(date +%s)" >> "$COUNTER"
n=$(wc -l < "$COUNTER")
if [ "$n" -gt "$MAX" ]; then
  echo "live claude wrapper: MAX_LIVE_SESSIONS ($MAX) exceeded" >&2
  echo "over $n" >> "$COUNTER.refused"
  exit 97
fi
if [ -n "$print" ] && [ -z "$budget" ]; then
  exec "$REAL" --max-budget-usd {run_usd} "$@"
fi
exec "$REAL" "$@"
"""


def write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def write_claude_wrapper(
    path: Path, real: str, counter: Path, maximum: int, run_usd: str = DEFAULT_RUN_USD
) -> None:
    write_executable(
        path,
        WRAPPER.format(
            real=shlex.quote(real),
            counter=shlex.quote(str(counter)),
            maximum=int(maximum),
            run_usd=shlex.quote(run_usd),
        ),
    )


def counted_sessions(counter: Path) -> int:
    try:
        return len([line for line in counter.read_text().splitlines() if line.strip()])
    except FileNotFoundError:
        return 0


def refusals(counter: Path) -> list[str]:
    refused = Path(f"{counter}.refused")
    try:
        return [line for line in refused.read_text().splitlines() if line.strip()]
    except FileNotFoundError:
        return []


class SessionBudget:
    """The suite's session counter: the counter file the wrappers append to, and the cap."""

    def __init__(self, counter: Path, maximum: int) -> None:
        self.counter = counter
        self.maximum = maximum

    @property
    def used(self) -> int:
        return counted_sessions(self.counter)

    def check_start(self, scenario: str, expected: int) -> None:
        """Fail (never skip) a billed test that could not finish inside the cap."""
        if self.used + expected > self.maximum:
            raise LiveError(
                f"{scenario}: would exceed MAX_LIVE_SESSIONS: {self.used} used + {expected} "
                f"expected > {self.maximum} (raise the variable or select fewer scenarios)"
            )

    def check_end(self, scenario: str, expected: int, used_before: int) -> list[str]:
        """Problems after a test: refusals by the wrapper, more sessions than registered."""
        problems = [f"{scenario}: the claude wrapper refused a launch: {r}" for r in refusals(self.counter)]
        took = self.used - used_before
        if took > expected:
            problems.append(
                f"{scenario}: started {took} claude sessions, the registry says {expected}"
            )
        return problems


# --- the fake crontab ----------------------------------------------------------------------

FAKE_CRONTAB = """#!/bin/sh
# A fake crontab: the table is the file {table}. Never reads or writes a real crontab.
T={table}
case "$1" in
  -l) if [ -f "$T" ]; then cat "$T"; else echo "no crontab for $(id -un)" >&2; exit 1; fi ;;
  -r) rm -f "$T" ;;
  -) cat > "$T" ;;
  *) echo "fake crontab: unsupported: $*" >&2; exit 2 ;;
esac
"""

TRIPWIRE_CRONTAB = """#!/bin/sh
echo 'crontab: a live test reached the crontab (tripwire stub)' >&2
exit 70
"""


def write_fake_crontab(path: Path, table: Path) -> None:
    write_executable(path, FAKE_CRONTAB.format(table=shlex.quote(str(table))))


def write_tripwire_crontab(path: Path) -> None:
    write_executable(path, TRIPWIRE_CRONTAB)


# --- a stub claude (for unbilled scenarios) ------------------------------------------------

STUB_CLAUDE = """#!/bin/sh
# A stub named claude: NOT Claude Code. Unbilled scenarios use it where a program called claude
# must exist (claude-session runs it, finds no hook event and ends the run on its exit).
case "$1" in
  --version|-v) echo "2.1.328 (Claude Code)"; exit 0 ;;
esac
echo "$$" >> {pidfile}
{body}
"""


def write_stub_claude(path: Path, pidfile: Path, body: str = "exit 3") -> None:
    write_executable(path, STUB_CLAUDE.format(pidfile=shlex.quote(str(pidfile)), body=body))


# --- processes -----------------------------------------------------------------------------


def run_process(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout: float,
    cwd: str | Path | None = None,
    input_text: str | None = None,
    new_session: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` to completion in its own session (process group). On a timeout exactly the
    group this call created is killed (its leader PID was recorded at launch and is still our
    unreaped child, so the PID cannot have been reused)."""
    child = subprocess.Popen(
        list(argv),
        env=dict(env),
        cwd=cwd,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=new_session,
    )
    try:
        out, err = child.communicate(input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(child)
        out, err = child.communicate()
        raise LiveError(
            f"timed out after {timeout:.0f}s: {' '.join(argv[:4])}...\nstdout: {out[-500:]}\n"
            f"stderr: {err[-500:]}"
        ) from None
    except BaseException:
        _kill_group(child)
        child.wait()
        raise
    return subprocess.CompletedProcess(list(argv), child.returncode, out, err)


def _kill_group(child: subprocess.Popen[str]) -> None:
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            child.kill()


def wait_for(
    predicate: Callable[[], Any], timeout: float, what: str, interval: float = 0.1
) -> Any:
    """Poll ``predicate`` until it returns a truthy value; fail with ``what`` after ``timeout``."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise LiveError(f"timed out after {timeout:.0f}s waiting for {what}")
        time.sleep(interval)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat_line.rsplit(")", 1)[-1].split()[0] != "Z"


# --- the World -----------------------------------------------------------------------------


def new_root() -> Path:
    """A fresh private root: short, directly in ``/tmp``, mode 0700, with the directories the
    tools expect (``TMUX_TMPDIR`` must exist and be private)."""
    root = Path(tempfile.mkdtemp(prefix="as-live-", dir="/tmp"))
    root.chmod(0o700)
    for name in (
        "home/.config",
        "home/.claude",
        "home/.cache",
        "home/.local/share",
        "home/.local/state",
        "tmp",
        "tmuxdir",
        "bin",
        "repos",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "tmuxdir").chmod(0o700)
    return root


@dataclass
class Repo:
    """A local git repository set up as a wtctl repo (``repo_id``): ``origin`` is a bare repo,
    ``clone`` the checkout wtctl resolves, ``sha`` the first commit."""

    repo_id: str
    origin: Path
    clone: Path
    sha: str


class World:
    """One test's private world. See the module docstring for the isolation rules."""

    def __init__(
        self,
        root: Path,
        tools: Mapping[str, Tool | None],
        environ: Mapping[str, str],
        *,
        budget: SessionBudget | None = None,
        auth: Auth | None = None,
        login: LoginCache | None = None,
        real_home: Path = Path("/nonexistent"),
        deadline: float | None = None,
    ) -> None:
        self.root = root
        self.tools = dict(tools)
        self.environ = dict(environ)
        self.budget = budget
        self.auth = auth
        self.login = login
        self.real_home = real_home
        self.deadline = deadline
        self.home = root / "home"
        self.state = self.home / ".local" / "state"
        self.config_home = self.home / ".config"
        self.tmuxdir = root / "tmuxdir"
        self.bin = root / "bin"
        self.crontab_table = root / "crontab.table"
        self.claude_pids = root / "stub-claude.pids"
        self.pids: list[tuple[int, str]] = []  # (pid, start token) of processes we may signal
        self.observations: list[tuple[str, Any]] = []
        self._installed = False
        self._stub = False
        write_tripwire_crontab(self.bin / "crontab")
        (self.home / ".gitconfig").write_text(
            "[user]\n\tname = Live Test\n\temail = live@example.invalid\n"
            "[init]\n\tdefaultBranch = main\n[commit]\n\tgpgsign = false\n"
        )
        self.link_tools()

    # -- tools and wrappers -----------------------------------------------------------------

    def link_tools(self) -> None:
        for name, tool in self.tools.items():
            if tool is None or name in ("git", "claude"):
                continue  # git comes from the standard PATH; claude is the counting wrapper
            link = self.bin / name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(tool.path)

    def install_claude(self, *, stub_body: str | None = None) -> None:
        """Put ``claude`` first on ``PATH``: the counting wrapper around the real binary, or
        with ``stub_body`` a stub that is not Claude Code (unbilled scenarios)."""
        link = self.bin / "claude"
        if stub_body is not None:
            write_stub_claude(link, self.claude_pids, stub_body)
            self._stub = True
            return
        tool = self.tools.get("claude")
        assert tool is not None and self.budget is not None
        write_claude_wrapper(
            link,
            tool.path,
            self.budget.counter,
            self.budget.maximum,
            self.environ.get(RUN_USD_ENV, DEFAULT_RUN_USD),
        )

    @property
    def claude_path(self) -> str:
        return str(self.bin / "claude")

    def install_auth(self) -> None:
        """Make the private home able to log in (billed tests only)."""
        assert self.auth is not None
        if self.login is not None:
            self.login.install(self.home)
        self._installed = True

    # -- environment and commands -----------------------------------------------------------

    def env(
        self,
        extra: Mapping[str, str] | None = None,
        *,
        auth: bool = False,
        tmux: Path | None = None,
    ) -> dict[str, str]:
        """The child environment. ``tmux`` is a private server socket to name in ``TMUX`` (as if
        the child ran inside that server); a socket outside the private root is refused."""
        env = build_env(
            self.environ,
            root=self.root,
            auth=self.auth if auth else None,
            extra=extra,
        )
        if tmux is not None:
            env["TMUX"] = f"{tmux},1,0"
        problems = environment_leaks(env, self.real_home, self.root)
        if problems:
            raise LiveError("private environment leaks: " + "; ".join(problems))
        return env

    def timeout(self, wanted: float) -> float:
        if self.deadline is None:
            return wanted
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise LiveTimeout("the test's wall-clock bound ran out")
        return min(wanted, left)

    def run(
        self,
        argv: Sequence[str],
        *,
        extra: Mapping[str, str] | None = None,
        auth: bool = False,
        timeout: float = 60,
        cwd: str | Path | None = None,
        input_text: str | None = None,
        tmux: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return run_process(
            argv,
            env=self.env(extra, auth=auth, tmux=tmux),
            timeout=self.timeout(timeout),
            cwd=cwd,
            input_text=input_text,
        )

    def tool(self, name: str, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return self.run([str(self.bin / name), *args], **kwargs)

    def json_tool(
        self, name: str, *args: str, ok: Sequence[int] = (0,), **kwargs: Any
    ) -> Any:
        # claude-session takes global flags after the command; termctl's `open ... -- argv` ends
        # its options at `--`, so its flag goes first.
        argv = [*args, "--json"] if name == "claude-session" else ["--json", *args]
        done = self.tool(name, *argv, **kwargs)
        return parse_json(done, ok, f"{name} {' '.join(args[:3])}")

    def tmux(self, *args: str, socket: str | Path | None = None, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """The real tmux against the private server (``TMUX_TMPDIR``'s default socket, or an
        explicit private socket path)."""
        tool = self.tools["tmux"]
        assert tool is not None
        sock = ["-S", str(socket)] if socket else []
        return self.run([tool.path, "-f", "/dev/null", *sock, *args], **kwargs)

    @property
    def default_socket(self) -> Path:
        return self.tmuxdir / f"tmux-{os.getuid()}" / "default"

    def stop_tmux(self) -> None:
        """Stop the private default server and any private socket under the root."""
        sockets = [self.default_socket, *self.root.glob("*.sock")]
        for sock in sockets:
            if sock.exists():
                try:
                    self.tmux("kill-server", socket=sock, timeout=10)
                except (LiveError, LiveTimeout, OSError):
                    pass

    # -- configuration files ----------------------------------------------------------------

    def write_config(self, tool: str, text: str, name: str = "config.yaml") -> Path:
        path = self.config_home / tool / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        path.write_text(text)
        path.chmod(0o600)
        return path

    def claude_session_config(self, extra: str = "") -> Path:
        auth_env = ""
        if self.auth is not None and "CLAUDE_CODE_OAUTH_TOKEN" in self.auth.env:
            auth_env = "claude_auth_env: [CLAUDE_CODE_OAUTH_TOKEN]\n"
        return self.write_config(
            "claude-session",
            f"claude_bin: {self.claude_path}\n{auth_env}{extra}",
        )

    def trust(self, *paths: Path) -> None:
        """Mark directories trusted in the private ``~/.claude.json`` (and approve the key)."""
        api_key = self.auth.api_key if self.auth else None
        path = self.home / ".claude.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(claude_json(list(paths), api_key=api_key), handle)

    def wtctl_config(self, repos: Mapping[str, Repo], extra: str = "") -> Path:
        lines = [f"root: {self.root / 'wtctl'}\n", "repos:\n"]
        for repo in repos.values():
            lines += [
                f"  {repo.repo_id}:\n",
                f"    clone: {repo.clone}\n",
                f"    clone_url: {repo.origin}\n",
            ]
        return self.write_config("wtctl", "".join(lines) + extra)

    # -- git repositories -------------------------------------------------------------------

    def git(self, *args: str, cwd: Path | None = None) -> str:
        done = self.run(["git", *args], cwd=cwd, timeout=60)
        if done.returncode != 0:
            raise LiveError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
        return done.stdout.strip()

    def make_repo(self, name: str = "proj", *, repo_id: str | None = None) -> Repo:
        """A bare origin plus a clone with one commit, wired as a wtctl repo id."""
        repo_id = repo_id or f"live.example/team/{name}"
        origin = self.root / "repos" / f"{name}-origin.git"
        clone = self.root / "repos" / name
        self.git("init", "--bare", "-q", str(origin))
        self.git("clone", "-q", str(origin), str(clone))
        (clone / "README.md").write_text("live test repository\n")
        self.git("add", "README.md", cwd=clone)
        self.git("commit", "-q", "-m", "first", cwd=clone)
        self.git("branch", "-M", "main", cwd=clone)
        self.git("push", "-q", "-u", "origin", "main", cwd=clone)
        self.git("--git-dir", str(origin), "symbolic-ref", "HEAD", "refs/heads/main")
        sha = self.git("rev-parse", "HEAD", cwd=clone)
        return Repo(repo_id, origin, clone, sha)

    def track(self, pid: int, *, tree: bool = False) -> None:
        """Remember a process (and with ``tree`` its current descendants) to kill at teardown, by
        PID plus start-time token, so one the test left behind never outlives it."""
        for one in [pid, *(descendant_pids(pid) if tree else [])]:
            token = process_token(one)
            if token is not None:
                self.pids.append((one, token))

    def observe(self, key: str, value: Any) -> None:
        """Record an observed (not asserted) value: printed in the summary and written to
        ``LIVE_OBSERVATIONS``; use it to update FIDELITY.md and design.md after a run."""
        self.observations.append((key, value))

    # -- teardown ---------------------------------------------------------------------------

    def close(self) -> None:
        self.stop_tmux()
        for pid, token in reversed(self.pids):
            if process_token(pid) == token:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if self.login is not None:
            self.login.collect(self.home)
        shutil.rmtree(self.root, ignore_errors=True)


def descendant_pids(pid: int) -> list[int]:
    """Current descendants of ``pid`` (Linux ``/proc`` walk)."""
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                parents[int(entry.name)] = int((entry / "stat").read_text().rsplit(")", 1)[-1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
    found: list[int] = []
    frontier = [pid]
    while frontier:
        parent = frontier.pop()
        for child, ppid in parents.items():
            if ppid == parent and child not in found:
                found.append(child)
                frontier.append(child)
    return found


def process_token(pid: int) -> str | None:
    """The start-time field of ``/proc/<pid>/stat``: with the PID it identifies a process."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return text.rsplit(")", 1)[-1].split()[19]


def parse_json(done: subprocess.CompletedProcess[str], ok: Sequence[int], what: str) -> Any:
    if done.returncode not in ok:
        raise LiveError(
            f"{what}: exit {done.returncode}, expected {list(ok)}\nstdout: {done.stdout[-600:]}\n"
            f"stderr: {done.stderr[-600:]}"
        )
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        raise LiveError(f"{what}: stdout is not JSON ({exc}): {done.stdout[-300:]!r}") from None


def check(condition: object, assumption: str, where: str, detail: object = "") -> None:
    """A live assertion whose failure names the assumption and the doc line that states it."""
    if not condition:
        raise LiveError(f"assumption broken: {assumption} [{where}] {detail}")
