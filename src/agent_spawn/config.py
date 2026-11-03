"""Stage 0 config: YAML file plus ``config.d`` drop-ins, validated into ``Config``.

Layering (interface.md, Config): built-in defaults < ``config.yaml`` < ``config.d/*.yaml`` in
lexical order < ``AGENT_SPAWN_*`` environment (four top-level scalar keys only). Maps merge,
``hooks.<hook>`` lists append, every other value (lists included) is replaced. Flags and per-repo
overrides sit on top: ``Config.resolve_promote`` applies them.

Every failure is ``config_invalid`` (exit 2) naming the file and the key. The loader reads files
only through ``safe_load`` rules: no anchors or aliases (no expansion bombs), no merge keys, no
duplicate or non-string keys, a size cap, regular files only. Messages never echo values.
"""

from __future__ import annotations

import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from .output import CliError

MAX_FILE_BYTES = 1024 * 1024
MAX_DROPINS = 256
DROPIN_DIR = "config.d"

# --- grammars (shape only) ------------------------------------------------------------------
# TODO(#4): the duration, kind, label and repo-id grammars are owned by validate.py/labels.py;
# these private copies mirror schemas/defs.json until those modules land.

_DURATION = re.compile(r"(?P<n>[0-9]+(?:\.[0-9]+)?)(?P<unit>ms|s|m|h|d)?")
_UNIT_SECONDS = {None: 1.0, "ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_KIND = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_LABEL_KEY = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_LABEL = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}=[A-Za-z0-9._:/#@+=,-]{0,256}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ME_NAME = re.compile(r"[A-Za-z0-9._@-]{1,64}")
_REPO_ID = re.compile(
    r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*(?::[0-9]{1,5})?"
    r"(?:/(?!\.\.?(?:/|$))[A-Za-z0-9._-]+)+"
)


def _pattern(regex: re.Pattern[str], what: str) -> Any:
    def check(value: str) -> str:
        if not regex.fullmatch(value):
            raise ValueError(f"not a valid {what}")
        return value

    return AfterValidator(check)


def parse_duration(value: object) -> float:
    """Seconds from a positive number, ``<number><unit>`` or a bare number string; zero refused."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError("a duration is a number of seconds or a string like 30s, 10m or 90")
    if isinstance(value, str):
        match = _DURATION.fullmatch(value)
        if match is None:
            raise ValueError("a duration is a number of seconds or a string like 30s, 10m or 90")
        try:
            seconds = float(match["n"]) * _UNIT_SECONDS[match["unit"]]
        except OverflowError:
            raise ValueError("duration out of range") from None
    else:
        seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("a duration must be positive and finite (zero is refused)")
    return seconds


Duration = Annotated[float, BeforeValidator(parse_duration)]
Kind = Annotated[str, _pattern(_KIND, "kind name")]
RepoId = Annotated[str, _pattern(_REPO_ID, "repo id")]
LabelKey = Annotated[str, _pattern(_LABEL_KEY, "label key")]
Label = Annotated[str, _pattern(_LABEL, "label (key=value)")]
EnvName = Annotated[str, _pattern(_ENV_NAME, "environment variable name")]

# --- models ---------------------------------------------------------------------------------

_STATES = (
    "starting working idle needs-input reported exited exited-unreported killed timeout lost "
    "not-started"
).split()
_STATUSES = "launching live parked tearing_down torn_down".split()


def _argv_item(value: str) -> str:
    if not value or "\0" in value:
        raise ValueError("argv elements must be non-empty and contain no NUL")
    return value


Argv = Annotated[list[Annotated[str, AfterValidator(_argv_item)]], Field(min_length=1)]


def _one_of(allowed: list[str]) -> Any:
    def check(value: str) -> str:
        if value not in allowed:
            raise ValueError(f"must be one of: {', '.join(allowed)}")
        return value

    return AfterValidator(check)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class Me(_Model):
    name: Annotated[str, _pattern(_ME_NAME, "user name")] | None = None


class Prune(_Model):
    older_than: Duration | None = None
    protect_labels: list[LabelKey | Label] = ["owner"]


class KindSettings(_Model):
    promote: bool = False


class RepoKindSettings(_Model):
    promote: bool | None = None


class RepoSettings(_Model):
    kinds: dict[Kind, RepoKindSettings] = {}


class PreHook(_Model):
    cmd: str = Field(min_length=1)
    timeout: Duration = 30.0
    kinds: list[Kind] | None = None
    exclude_kinds: list[Kind] | None = None
    labels: list[Label] | None = None

    @field_validator("cmd")
    @classmethod
    def _no_nul(cls, value: str) -> str:
        if "\0" in value:
            raise ValueError("cmd must not contain NUL")
        return value


class PostHook(PreHook):
    async_: bool = Field(default=False, alias="async")


class StateChangeHook(PostHook):
    states: list[Annotated[str, _one_of(_STATES)]] | None = None
    statuses: list[Annotated[str, _one_of(_STATUSES)]] | None = None


class Hooks(_Model):
    pre_launch: list[PreHook] = Field(default=[], alias="pre-launch")
    pre_teardown: list[PreHook] = Field(default=[], alias="pre-teardown")
    post_launch: list[PostHook] = Field(default=[], alias="post-launch")
    post_state_change: list[StateChangeHook] = Field(default=[], alias="post-state-change")
    post_escalate: list[PostHook] = Field(default=[], alias="post-escalate")
    post_park: list[PostHook] = Field(default=[], alias="post-park")
    post_teardown: list[PostHook] = Field(default=[], alias="post-teardown")


class Config(_Model):
    me: Me = Field(default_factory=Me)
    claude_session_cmd: Argv = ["claude-session"]
    termctl_cmd: Argv = ["termctl"]
    wtctl_cmd: Argv | None = ["wtctl"]
    job_id_env: EnvName = "JOBSCTL_JOB_ID"
    self_label_keys: list[LabelKey] = ["pr"]
    handoff_timeout: Duration = 30.0
    auto_report_resume: bool = True
    tombstone_days: int = Field(default=30, ge=0)
    prune: Prune = Field(default_factory=Prune)
    kinds: dict[Kind, KindSettings] = {}
    repos: dict[RepoId, RepoSettings] = {}
    hooks: Hooks = Field(default_factory=Hooks)

    @field_validator("self_label_keys")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("keys must be unique")
        return value

    def resolve_promote(self, *, kind: str | None, repo: str | None, explicit: bool | None) -> bool:
        """flags > per-repo override > kinds.<kind>.promote > false (interface.md, Promotion)."""
        if explicit is not None:
            return explicit
        if kind is not None:
            if repo is not None and repo in self.repos:
                override = self.repos[repo].kinds.get(kind)
                if override is not None and override.promote is not None:
                    return override.promote
            if kind in self.kinds:
                return self.kinds[kind].promote
        return False


# --- environment overrides ------------------------------------------------------------------

_ENV_INT = re.compile(r"[0-9]{1,9}")


def _env_bool(value: str) -> bool:
    if value in ("true", "false"):
        return value == "true"
    raise ValueError("expected true or false")


def _env_int(value: str) -> int:
    if not _ENV_INT.fullmatch(value):
        raise ValueError("expected a non-negative integer")
    return int(value)


def _env_text(value: str) -> str:
    return value


# key -> (variable, converter); durations stay strings for the model's own parser.
ENV_OVERRIDES: dict[str, tuple[str, Any]] = {
    "handoff_timeout": ("AGENT_SPAWN_HANDOFF_TIMEOUT", _env_text),
    "tombstone_days": ("AGENT_SPAWN_TOMBSTONE_DAYS", _env_int),
    "job_id_env": ("AGENT_SPAWN_JOB_ID_ENV", _env_text),
    "auto_report_resume": ("AGENT_SPAWN_AUTO_REPORT_RESUME", _env_bool),
}

# --- reading files --------------------------------------------------------------------------


class _Refused(yaml.YAMLError):
    pass


class _SafeConfigLoader(yaml.SafeLoader):
    """SafeLoader without anchors, aliases, merge keys, duplicate keys or non-string keys."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise _Refused(f"line {self.peek_event().start_mark.line + 1}: aliases are not allowed")
        event = self.peek_event()
        if getattr(event, "anchor", None) is not None:
            raise _Refused(f"line {event.start_mark.line + 1}: anchors are not allowed")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        seen: set[str] = set()
        for key_node, _ in node.value:
            line = key_node.start_mark.line + 1
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise _Refused(f"line {line}: merge keys are not allowed")
            if key_node.tag != "tag:yaml.org,2002:str":
                raise _Refused(f"line {line}: mapping keys must be strings")
            key = str(key_node.value)
            if key in seen:
                raise _Refused(f"line {line}: duplicate key {_clip(key)!r}")
            seen.add(key)
        return super().construct_mapping(node, deep)


def _clip(text: str) -> str:
    return text if len(text) <= 64 else text[:61] + "..."


def _fail(message: str) -> CliError:
    return CliError.of("config_invalid", message)


def _read_regular(path: str) -> bytes | None:
    """The file's bytes; ``None`` when it does not exist. Symlinks are followed."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _fail(f"cannot read config file {path}: {exc.strerror or 'error'}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise _fail(f"config file {path} is not a regular file")
        if info.st_size > MAX_FILE_BYTES:
            raise _fail(f"config file {path} is larger than {MAX_FILE_BYTES} bytes")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                raise _fail(f"config file {path} is larger than {MAX_FILE_BYTES} bytes")
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise _fail(f"cannot read config file {path}: {exc.strerror or 'error'}") from None
    finally:
        os.close(fd)


def _parse(path: str, data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise _fail(f"config file {path} is not valid UTF-8") from None
    try:
        root = yaml.load(text, Loader=_SafeConfigLoader)  # noqa: S506  (safe subclass)
    except _Refused as exc:
        raise _fail(f"config file {path}: {exc}") from None
    except yaml.MarkedYAMLError as exc:
        line = exc.problem_mark.line + 1 if exc.problem_mark is not None else 0
        raise _fail(f"config file {path}: invalid YAML at line {line}: {exc.problem}") from None
    except yaml.YAMLError:
        raise _fail(f"config file {path}: invalid YAML") from None
    except RecursionError:
        raise _fail(f"config file {path}: nested too deeply") from None
    if root is None:
        return {}
    if not isinstance(root, dict):
        raise _fail(f"config file {path}: the top level must be a mapping")
    return root  # pyright: ignore[reportUnknownVariableType]


def _merge(base: dict[str, Any], over: Mapping[str, Any], path: tuple[str, ...] = ()) -> None:
    for key, value in over.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _merge(current, value, (*path, key))  # pyright: ignore[reportUnknownArgumentType]
        elif path == ("hooks",) and isinstance(current, list) and isinstance(value, list):
            current.extend(value)  # pyright: ignore[reportUnknownMemberType]
        else:
            base[key] = value


def _describe(error: ValidationError, where: str, env_keys: Mapping[str, str] | None = None) -> str:
    parts: list[str] = []
    for item in error.errors(include_input=False, include_url=False)[:5]:
        loc = [str(p) for p in item["loc"] if p != "[key]"]
        key = ".".join(loc) or "(root)"
        origin = f" (from {env_keys[loc[0]]})" if env_keys and loc and loc[0] in env_keys else ""
        msg = item["msg"].removeprefix("Value error, ")
        parts.append(f"key `{key}`{origin}: {msg}")
    return f"invalid config {where}: " + "; ".join(parts)


def _validate(raw: dict[str, Any], where: str, env_keys: Mapping[str, str] | None = None) -> Config:
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise _fail(_describe(exc, where, env_keys)) from None


# --- public API -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedConfig:
    """``path`` is the absolute config path to store and pass to internal processes."""

    config: Config
    path: str
    files: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def default_config_path(environ: Mapping[str, str]) -> str:
    base = environ.get("XDG_CONFIG_HOME", "")
    if not base.startswith("/"):
        home = environ.get("HOME", "")
        home = home if home.startswith("/") else os.path.expanduser("~")
        base = os.path.join(home, ".config")
    return os.path.join(base, "agent-spawn", "config.yaml")


def _dropins(directory: str) -> list[str]:
    try:
        with os.scandir(directory) as entries:
            names = sorted(
                e.name for e in entries if e.name.endswith(".yaml") and not e.name.startswith(".")
            )
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise _fail(f"cannot read {directory}: {exc.strerror or 'error'}") from None
    if len(names) > MAX_DROPINS:
        raise _fail(f"{directory} has more than {MAX_DROPINS} drop-in files")
    return [os.path.join(directory, name) for name in names]


def load_config(
    path: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    warn_missing: bool = False,
) -> LoadedConfig:
    """Load and validate the config; raises ``CliError`` ``config_invalid`` (exit 2).

    ``path`` is the config file (default: ``$XDG_CONFIG_HOME/agent-spawn/config.yaml``). Drop-ins
    are ``config.d/*.yaml`` beside it. A missing file is the built-in defaults (``warn_missing``
    adds a warning, for a path the user named).
    """
    env = os.environ if environ is None else environ
    if path is not None and "\0" in path:
        raise _fail("the config path contains a NUL byte")
    absolute = os.path.abspath(path) if path is not None else default_config_path(env)
    warnings: list[str] = []
    files: list[str] = []
    merged: dict[str, Any] = {}

    main = _read_regular(absolute)
    if main is None:
        if warn_missing:
            warnings.append(f"config file {absolute} does not exist; using built-in defaults")
    else:
        files.append(absolute)
    sources: list[tuple[str, bytes]] = [] if main is None else [(absolute, main)]
    for dropin in _dropins(os.path.join(os.path.dirname(absolute), DROPIN_DIR)):
        data = _read_regular(dropin)
        if data is None:
            raise _fail(f"config drop-in {dropin} does not exist (dangling link?)")
        files.append(dropin)
        sources.append((dropin, data))

    for source, data in sources:
        raw = _parse(source, data)
        _validate(raw, f"in {source}")  # attribute errors to the file
        _merge(merged, raw)

    env_keys: dict[str, str] = {}
    for key, (variable, convert) in ENV_OVERRIDES.items():
        if variable not in env:
            continue
        try:
            merged[key] = convert(env[variable])
        except ValueError as exc:
            raise _fail(f"invalid environment variable {variable}: {exc}") from None
        env_keys[key] = variable

    config = _validate(merged, "after merging", env_keys)
    return LoadedConfig(config, absolute, tuple(files), tuple(warnings))
