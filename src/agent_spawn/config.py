"""Stage 0 config: YAML file plus ``config.d`` drop-ins, validated into ``Config``.

Layering (interface.md, Config): built-in defaults < ``config.yaml`` < ``config.d/*.yaml`` in
lexical order < ``AGENT_SPAWN_*`` environment (four top-level scalar keys only). Maps merge,
``hooks.<hook>`` lists append, every other value (lists included) is replaced. Flags and per-repo
overrides sit on top: ``Config.resolve_promote`` applies them.

Every failure is ``config_invalid`` (exit 2) naming the file and the key. The loader reads files
with YAML 1.2 core-schema scalars on a safe loader: no anchors or aliases (no expansion bombs), no
merge keys, no explicit tags, no duplicate or non-string keys, a size cap, regular files only.
Messages never echo values.
"""

from __future__ import annotations

import math
import os
import pwd
import re
import stat
from collections.abc import Mapping, Sequence
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
        number: str | float = match["n"]
        unit = _UNIT_SECONDS[match["unit"]]
    else:
        number, unit = value, 1.0
    try:
        seconds = float(number) * unit
    except OverflowError:
        raise ValueError("duration out of range") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("a duration must be positive and finite (zero is refused)")
    return seconds


Duration = Annotated[float, BeforeValidator(parse_duration)]
Kind = Annotated[str, _pattern(_KIND, "kind name")]
RepoId = Annotated[str, _pattern(_REPO_ID, "repo id")]
LabelKey = Annotated[str, _pattern(_LABEL_KEY, "label key")]
Label = Annotated[str, _pattern(_LABEL, "label (key=value)")]


def _label_key_or_label(value: str) -> str:
    if not (_LABEL_KEY.fullmatch(value) or _LABEL.fullmatch(value)):
        raise ValueError("not a valid label key or label")
    return value


LabelKeyOrLabel = Annotated[str, AfterValidator(_label_key_or_label)]


def _refuse_null(value: object) -> object:
    if value is None:
        raise ValueError("must not be null (omit the key instead)")
    return value


# An optional key whose schema type has no null: absent is the default, an explicit null is refused.
NotNull = BeforeValidator(_refuse_null)
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


def _one_of(allowed: Sequence[str]) -> Any:
    def check(value: str) -> str:
        if value not in allowed:
            raise ValueError(f"must be one of: {', '.join(allowed)}")
        return value

    return AfterValidator(check)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Me(_Model):
    name: Annotated[str, _pattern(_ME_NAME, "user name")] | None = None


class Prune(_Model):
    older_than: Duration | None = None
    protect_labels: list[LabelKeyOrLabel] = ["owner"]


class KindSettings(_Model):
    promote: bool = False


class RepoKindSettings(_Model):
    promote: Annotated[bool | None, NotNull] = None


class RepoSettings(_Model):
    kinds: dict[Kind, RepoKindSettings] = {}


class PreHook(_Model):
    cmd: str = Field(min_length=1)
    timeout: Duration = 30.0
    kinds: Annotated[list[Kind] | None, NotNull] = None
    exclude_kinds: Annotated[list[Kind] | None, NotNull] = None
    labels: Annotated[list[Label] | None, NotNull] = None

    @field_validator("cmd")
    @classmethod
    def _no_nul(cls, value: str) -> str:
        if "\0" in value:
            raise ValueError("cmd must not contain NUL")
        return value


class PostHook(PreHook):
    async_: bool = Field(default=False, alias="async")


class StateChangeHook(PostHook):
    states: Annotated[list[Annotated[str, _one_of(_STATES)]] | None, NotNull] = None
    statuses: Annotated[list[Annotated[str, _one_of(_STATUSES)]] | None, NotNull] = None


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
    """SafeLoader with YAML 1.2 core-schema scalars; no anchors, aliases, tags or duplicate keys.

    Only ``true``/``false`` (three spellings each) are booleans, only ``null``/``~``/empty is null,
    integers are decimal and floats the 1.2 core form; every other plain scalar (``yes``, ``on``,
    ``010`` as octal, ``0x10``, ``1:30``, ``1_0``, timestamps, ``<<``) is a string.
    """

    yaml_implicit_resolvers: dict[Any, Any] = {}

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise _Refused(f"line {self.peek_event().start_mark.line + 1}: aliases are not allowed")
        event = self.peek_event()
        if getattr(event, "anchor", None) is not None:
            raise _Refused(f"line {event.start_mark.line + 1}: anchors are not allowed")
        if getattr(event, "tag", None) is not None:
            raise _Refused(f"line {event.start_mark.line + 1}: explicit tags are not allowed")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        seen: set[str] = set()
        for key_node, _ in node.value:
            key = str(key_node.value)
            if key in seen:
                raise _Refused(f"line {key_node.start_mark.line + 1}: duplicate key {_clip(key)!r}")
            seen.add(key)
        return super().construct_mapping(node, deep)


def _core_resolver(tag: str, pattern: str, first: str) -> None:
    _SafeConfigLoader.add_implicit_resolver(
        f"tag:yaml.org,2002:{tag}", re.compile(pattern), list(first)
    )


# Order matters within a first character: int before float.
_core_resolver("bool", r"^(?:true|True|TRUE|false|False|FALSE)$", "tTfF")
_core_resolver("null", r"^(?:~|null|Null|NULL|)$", "~nN")
_SafeConfigLoader.add_implicit_resolver("tag:yaml.org,2002:null", re.compile(r"^$"), [""])
_core_resolver("int", r"^[-+]?[0-9]+$", "-+0123456789")
_core_resolver(
    "float",
    r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$",
    "-+.0123456789",
)


def _construct_decimal_int(loader: _SafeConfigLoader, node: Any) -> int:
    return int(loader.construct_scalar(node), 10)  # decimal only: 010 is ten


_SafeConfigLoader.add_constructor("tag:yaml.org,2002:int", _construct_decimal_int)


def _render_key(key: object) -> str:
    if key is None:
        return "null"
    if isinstance(key, bool):
        return "true" if key else "false"
    return _clip(str(key))


def _check_keys(node: object, path: tuple[str, ...] = ()) -> None:
    """Refuse non-string mapping keys, naming the key path (parent path plus the key)."""
    if isinstance(node, dict):
        for key, value in node.items():  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(key, str):
                where = ".".join((*path, _render_key(key)))
                raise _Refused(f"key `{where}`: mapping keys must be strings")
            _check_keys(value, (*path, key))
    elif isinstance(node, list):
        for i, item in enumerate(node):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            _check_keys(item, (*path, str(i)))


def _clip(text: str) -> str:
    return text if len(text) <= 64 else text[:61] + "..."


def _fail(message: str) -> CliError:
    return CliError.of("config_invalid", message)


def _dangling(path: str) -> bool:
    return os.path.islink(path) and not os.path.exists(path)


def _read_regular(path: str) -> bytes | None:
    """The file's bytes; ``None`` when it does not exist. Symlinks are followed."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        for candidate in (path, os.path.dirname(path)):
            if _dangling(candidate):
                raise _fail(f"config path {candidate} is a dangling symlink") from None
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
        _check_keys(root)
    except _Refused as exc:
        raise _fail(f"config file {path}: {exc}") from None
    except yaml.MarkedYAMLError as exc:
        line = exc.problem_mark.line + 1 if exc.problem_mark is not None else 0
        raise _fail(f"config file {path}: invalid YAML at line {line}: {exc.problem}") from None
    except yaml.YAMLError:
        raise _fail(f"config file {path}: invalid YAML") from None
    except RecursionError:
        raise _fail(f"config file {path}: nested too deeply") from None
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError):
        # e.g. an integer longer than Python's digit limit; the value is never echoed
        raise _fail(f"config file {path}: a scalar value cannot be converted") from None
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


def _passwd_home() -> str | None:
    try:
        return pwd.getpwuid(os.geteuid()).pw_dir
    except KeyError:
        return None


def default_config_path(environ: Mapping[str, str]) -> str:
    """``$XDG_CONFIG_HOME`` if absolute, else ``$HOME`` if absolute, else the passwd home, plus
    ``.config``; always absolute (``config_invalid`` when no absolute base exists)."""
    base = environ.get("XDG_CONFIG_HOME", "")
    if not base.startswith("/"):
        home = environ.get("HOME", "")
        if not home.startswith("/"):
            home = _passwd_home() or ""
        if not home.startswith("/"):
            raise _fail(
                "cannot find the default config path: neither XDG_CONFIG_HOME, HOME nor the "
                "passwd entry gives an absolute directory"
            )
        base = os.path.join(home, ".config")
    return os.path.join(base, "agent-spawn", "config.yaml")


def _dropins(directory: str) -> list[str]:
    try:
        with os.scandir(directory) as entries:
            names = sorted(
                e.name for e in entries if e.name.endswith(".yaml") and not e.name.startswith(".")
            )
    except FileNotFoundError:
        if _dangling(directory):
            raise _fail(f"config path {directory} is a dangling symlink") from None
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
    must_exist: bool = False,
) -> LoadedConfig:
    """Load and validate the config; raises ``CliError`` ``config_invalid`` (exit 2).

    ``path`` is the config file (default: ``default_config_path``). Drop-ins are
    ``config.d/*.yaml`` beside it. A missing file is the built-in defaults, unless ``must_exist``
    (a path the user named): then it is ``config_invalid``.
    """
    env = os.environ if environ is None else environ
    if path is None:
        absolute = default_config_path(env)
    else:
        if "\0" in path:
            raise _fail("the config path contains a NUL byte")
        try:
            absolute = os.path.abspath(path)
        except OSError as exc:
            reason = exc.strerror or "error"
            raise _fail(f"cannot resolve relative config path: {reason}") from None
    files: list[str] = []
    merged: dict[str, Any] = {}

    main = _read_regular(absolute)
    if main is None:
        if must_exist:
            raise _fail(f"config file {absolute} does not exist")
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
    return LoadedConfig(config, absolute, tuple(files))
