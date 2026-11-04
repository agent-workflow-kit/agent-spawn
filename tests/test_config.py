"""Config loading: defaults, drop-ins, overrides, validation, hostile YAML."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_spawn import config as config_module
from agent_spawn.config import Config, default_config_path, load_config
from agent_spawn.output import CliError

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "schemas" / "config.json").read_text())


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def load(tmp_path: Path, text: str | None = None, env: dict[str, str] | None = None) -> Config:
    cfg = tmp_path / "cfg" / "config.yaml"
    if text is not None:
        write(cfg, text)
    return load_config(str(cfg), environ=env or {}).config


def invalid(tmp_path: Path, text: str, env: dict[str, str] | None = None) -> str:
    with pytest.raises(CliError) as info:
        load(tmp_path, text, env)
    assert info.value.code == "config_invalid"
    assert info.value.exit_code == 2
    return info.value.message


# --- defaults ---------------------------------------------------------------------------------


def test_example_file_equals_the_builtin_defaults() -> None:
    example = yaml.safe_load((ROOT / "config.example.yaml").read_text())
    assert Config.model_validate(example) == Config()


def test_example_file_names_every_top_level_key() -> None:
    example = yaml.safe_load((ROOT / "config.example.yaml").read_text())
    assert set(example) == set(Config.model_fields)


def test_example_file_validates_against_the_schema() -> None:
    import jsonschema

    jsonschema.validate(yaml.safe_load((ROOT / "config.example.yaml").read_text()), SCHEMA)
    jsonschema.validate(Config().model_dump(mode="json", by_alias=True), SCHEMA)


@pytest.mark.parametrize(
    "doc",
    [
        {"claude_session_cmd": ["a\0b"]},
        {"termctl_cmd": ["a\0b"]},
        {"wtctl_cmd": ["a\0b"]},
        {"hooks": {"pre-launch": [{"cmd": "a\0b"}]}},
        {"hooks": {"post-launch": [{"cmd": "a\0b"}]}},
        {"hooks": {"post-state-change": [{"cmd": "a\0b"}]}},
        {"tombstone_days": 30.5},
    ],
)
def test_schema_refuses_what_the_model_refuses(doc: dict[str, Any]) -> None:
    import jsonschema

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, SCHEMA)
    with pytest.raises(Exception):  # noqa: B017, PT011  (pydantic ValidationError)
        Config.model_validate(doc)


def test_defaults_equal_the_schema_defaults() -> None:
    dumped = Config().model_dump(mode="json", by_alias=True)
    for key, spec in SCHEMA["properties"].items():
        if key == "handoff_timeout":
            assert dumped[key] == 30.0
        elif key in ("me", "prune", "hooks"):
            # nested defaults: the schema's per-property defaults
            for sub, sub_spec in spec["properties"].items():
                if "default" in sub_spec:
                    assert dumped[key][sub] == sub_spec["default"], f"{key}.{sub}"
        else:
            assert dumped[key] == spec["default"], key


def test_missing_file_means_defaults(tmp_path: Path) -> None:
    loaded = load_config(str(tmp_path / "nope.yaml"), environ={})
    assert loaded.config == Config()
    assert loaded.files == ()


def test_empty_file_means_defaults(tmp_path: Path) -> None:
    assert load(tmp_path, "") == Config()
    assert load(tmp_path, "# only a comment\n") == Config()


# --- paths --------------------------------------------------------------------------------------


def test_default_path_uses_xdg_then_home() -> None:
    assert default_config_path({"XDG_CONFIG_HOME": "/x", "HOME": "/h"}) == (
        "/x/agent-spawn/config.yaml"
    )
    assert default_config_path({"HOME": "/h"}) == "/h/.config/agent-spawn/config.yaml"
    # a relative XDG_CONFIG_HOME is ignored (XDG basedir rule)
    assert default_config_path({"XDG_CONFIG_HOME": "rel", "HOME": "/h"}) == (
        "/h/.config/agent-spawn/config.yaml"
    )


@pytest.mark.parametrize("home", [None, "", "rel/home"])
def test_default_path_falls_back_to_the_passwd_home(
    monkeypatch: pytest.MonkeyPatch, home: str | None
) -> None:
    monkeypatch.setattr(config_module, "_passwd_home", lambda: "/pw")
    env = {"XDG_CONFIG_HOME": "rel"} if home is None else {"HOME": home}
    assert default_config_path(env) == "/pw/.config/agent-spawn/config.yaml"


def test_default_path_ignores_the_process_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/process-home")
    monkeypatch.setattr(config_module, "_passwd_home", lambda: "/pw")
    assert default_config_path({}) == "/pw/.config/agent-spawn/config.yaml"


@pytest.mark.parametrize("passwd", [None, "", "relative"])
def test_default_path_without_any_absolute_home_is_invalid(
    monkeypatch: pytest.MonkeyPatch, passwd: str | None
) -> None:
    monkeypatch.setattr(config_module, "_passwd_home", lambda: passwd)
    with pytest.raises(CliError) as info:
        default_config_path({"HOME": "rel"})
    assert info.value.code == "config_invalid"


def test_relative_path_with_a_deleted_cwd_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    def gone() -> str:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(os, "getcwd", gone)
    with pytest.raises(CliError) as info:
        load_config("c.yaml", environ={})
    assert info.value.code == "config_invalid"
    assert "relative config path" in info.value.message


def test_stored_path_is_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(tmp_path / "c.yaml", "tombstone_days: 3\n")
    monkeypatch.chdir(tmp_path)
    loaded = load_config("c.yaml", environ={})
    assert loaded.path == str(tmp_path / "c.yaml")
    assert loaded.config.tombstone_days == 3


def test_none_path_loads_the_default_location(tmp_path: Path) -> None:
    write(tmp_path / "xdg" / "agent-spawn" / "config.yaml", "tombstone_days: 4\n")
    env = {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}
    loaded = load_config(None, environ=env)
    assert loaded.config.tombstone_days == 4
    assert loaded.path == str(tmp_path / "xdg" / "agent-spawn" / "config.yaml")


# --- drop-ins -----------------------------------------------------------------------------------


def test_dropins_merge_in_lexical_order(tmp_path: Path) -> None:
    base = write(
        tmp_path / "config.yaml", "tombstone_days: 1\nme: {name: a}\nhandoff_timeout: 5s\n"
    )
    write(tmp_path / "config.d" / "20-b.yaml", "tombstone_days: 2\n")
    write(tmp_path / "config.d" / "10-a.yaml", "tombstone_days: 9\nme: {name: z}\n")
    write(tmp_path / "config.d" / "ignored.txt", "tombstone_days: 77\n")
    loaded = load_config(str(base), environ={})
    assert loaded.config.tombstone_days == 2  # 10-a then 20-b
    assert loaded.config.me.name == "z"
    assert loaded.config.handoff_timeout == 5.0
    assert [os.path.basename(f) for f in loaded.files] == ["config.yaml", "10-a.yaml", "20-b.yaml"]


def test_maps_merge_and_other_lists_replace(tmp_path: Path) -> None:
    base = write(
        tmp_path / "config.yaml",
        "claude_session_cmd: [a, b]\nkinds: {x: {promote: true}}\nprune: {protect_labels: [p]}\n",
    )
    write(
        tmp_path / "config.d" / "1.yaml",
        "claude_session_cmd: [c]\nkinds: {y: {promote: true}}\nprune: {older_than: 2d}\n",
    )
    cfg = load_config(str(base), environ={}).config
    assert cfg.claude_session_cmd == ["c"]
    assert set(cfg.kinds) == {"x", "y"}
    assert cfg.prune.protect_labels == ["p"]
    assert cfg.prune.older_than == 2 * 86400


def test_hook_lists_append_across_files(tmp_path: Path) -> None:
    base = write(tmp_path / "config.yaml", "hooks:\n  post-launch:\n    - cmd: one\n")
    write(tmp_path / "config.d" / "1.yaml", "hooks:\n  post-launch:\n    - cmd: two\n")
    write(tmp_path / "config.d" / "2.yaml", "hooks:\n  pre-launch:\n    - cmd: three\n")
    cfg = load_config(str(base), environ={}).config
    assert [h.cmd for h in cfg.hooks.post_launch] == ["one", "two"]
    assert [h.cmd for h in cfg.hooks.pre_launch] == ["three"]


def test_dropin_null_replaces_wtctl_cmd(tmp_path: Path) -> None:
    base = write(tmp_path / "config.yaml", "wtctl_cmd: [x]\n")
    write(tmp_path / "config.d" / "1.yaml", "wtctl_cmd: null\n")
    assert load_config(str(base), environ={}).config.wtctl_cmd is None


def test_dropin_error_names_file_and_key(tmp_path: Path) -> None:
    base = write(tmp_path / "config.yaml", "")
    write(tmp_path / "config.d" / "5-bad.yaml", "tombstone_days: -1\n")
    with pytest.raises(CliError) as info:
        load_config(str(base), environ={})
    assert "5-bad.yaml" in info.value.message
    assert "tombstone_days" in info.value.message


def test_dropin_dir_that_is_a_file_is_invalid(tmp_path: Path) -> None:
    base = write(tmp_path / "config.yaml", "")
    write(tmp_path / "config.d", "x")
    with pytest.raises(CliError):
        load_config(str(base), environ={})


def test_dropin_that_is_a_directory_or_dangling_link_is_invalid(tmp_path: Path) -> None:
    base = write(tmp_path / "config.yaml", "")
    (tmp_path / "config.d").mkdir()
    (tmp_path / "config.d" / "a.yaml").mkdir()
    with pytest.raises(CliError):
        load_config(str(base), environ={})
    (tmp_path / "config.d" / "a.yaml").rmdir()
    (tmp_path / "config.d" / "b.yaml").symlink_to(tmp_path / "missing")
    with pytest.raises(CliError):
        load_config(str(base), environ={})


def test_dangling_config_file_link_is_invalid(tmp_path: Path) -> None:
    link = tmp_path / "config.yaml"
    link.symlink_to(tmp_path / "missing.yaml")
    with pytest.raises(CliError) as info:
        load_config(str(link), environ={})
    assert info.value.code == "config_invalid"
    assert str(link) in info.value.message


def test_dangling_config_dir_link_is_invalid(tmp_path: Path) -> None:
    (tmp_path / "xdg").mkdir()
    (tmp_path / "xdg" / "agent-spawn").symlink_to(tmp_path / "missing-dir")
    with pytest.raises(CliError) as info:
        load_config(None, environ={"XDG_CONFIG_HOME": str(tmp_path / "xdg")})
    assert info.value.code == "config_invalid"
    assert str(tmp_path / "xdg" / "agent-spawn") in info.value.message


def test_dangling_dropin_dir_link_is_invalid(tmp_path: Path) -> None:
    base = write(tmp_path / "config.yaml", "")
    (tmp_path / "config.d").symlink_to(tmp_path / "missing-dir")
    with pytest.raises(CliError) as info:
        load_config(str(base), environ={})
    assert str(tmp_path / "config.d") in info.value.message


def test_missing_file_must_exist_is_invalid(tmp_path: Path) -> None:
    with pytest.raises(CliError) as info:
        load_config(str(tmp_path / "gone.yaml"), environ={}, must_exist=True)
    assert info.value.code == "config_invalid"
    assert str(tmp_path / "gone.yaml") in info.value.message


def test_symlinked_config_file_is_followed(tmp_path: Path) -> None:
    real = write(tmp_path / "real.yaml", "tombstone_days: 8\n")
    link = tmp_path / "link.yaml"
    link.symlink_to(real)
    assert load_config(str(link), environ={}).config.tombstone_days == 8


# --- environment overrides ----------------------------------------------------------------------


def test_env_overrides_config(tmp_path: Path) -> None:
    text = "handoff_timeout: 5s\ntombstone_days: 1\njob_id_env: A\nauto_report_resume: true\n"
    cfg = load(
        tmp_path,
        text,
        {
            "AGENT_SPAWN_HANDOFF_TIMEOUT": "45s",
            "AGENT_SPAWN_TOMBSTONE_DAYS": "7",
            "AGENT_SPAWN_JOB_ID_ENV": "MY_JOB",
            "AGENT_SPAWN_AUTO_REPORT_RESUME": "false",
        },
    )
    assert (cfg.handoff_timeout, cfg.tombstone_days, cfg.job_id_env) == (45.0, 7, "MY_JOB")
    assert cfg.auto_report_resume is False


def test_env_bare_number_string_is_seconds(tmp_path: Path) -> None:
    assert load(tmp_path, None, {"AGENT_SPAWN_HANDOFF_TIMEOUT": "90"}).handoff_timeout == 90.0


def test_env_beats_dropins(tmp_path: Path) -> None:
    base = write(tmp_path / "config.yaml", "tombstone_days: 1\n")
    write(tmp_path / "config.d" / "1.yaml", "tombstone_days: 2\n")
    cfg = load_config(str(base), environ={"AGENT_SPAWN_TOMBSTONE_DAYS": "3"}).config
    assert cfg.tombstone_days == 3


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("AGENT_SPAWN_HANDOFF_TIMEOUT", "0"),
        ("AGENT_SPAWN_HANDOFF_TIMEOUT", "0s"),
        ("AGENT_SPAWN_HANDOFF_TIMEOUT", "soon"),
        ("AGENT_SPAWN_HANDOFF_TIMEOUT", ""),
        ("AGENT_SPAWN_TOMBSTONE_DAYS", "-1"),
        ("AGENT_SPAWN_TOMBSTONE_DAYS", "1.5"),
        ("AGENT_SPAWN_TOMBSTONE_DAYS", "x"),
        ("AGENT_SPAWN_JOB_ID_ENV", "1BAD"),
        ("AGENT_SPAWN_JOB_ID_ENV", "A=B"),
        ("AGENT_SPAWN_AUTO_REPORT_RESUME", "maybe"),
    ],
)
def test_invalid_env_names_the_variable(tmp_path: Path, var: str, value: str) -> None:
    message = invalid(tmp_path, "", {var: value})
    assert var in message


def test_other_env_vars_are_ignored(tmp_path: Path) -> None:
    env = {
        "AGENT_SPAWN_ID": "ag-x",
        "AGENT_SPAWN_HOOK": "h",
        "AGENT_SPAWN_PRUNE": "{}",
        "AGENT_SPAWN_KINDS": "x",
    }
    assert load(tmp_path, "", env) == Config()


# --- validation ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("bogus: 1\n", "bogus"),
        ("me: {nam: x}\n", "me.nam"),
        ("me: {name: 'a b'}\n", "me.name"),
        ("me: {name: ''}\n", "me.name"),
        ("claude_session_cmd: []\n", "claude_session_cmd"),
        ("claude_session_cmd: ['']\n", "claude_session_cmd.0"),
        ("claude_session_cmd: x\n", "claude_session_cmd"),
        ("termctl_cmd: null\n", "termctl_cmd"),
        ("job_id_env: 'a-b'\n", "job_id_env"),
        ("self_label_keys: [A]\n", "self_label_keys.0"),
        ("self_label_keys: [pr, pr]\n", "self_label_keys"),
        ("handoff_timeout: 0\n", "handoff_timeout"),
        ("handoff_timeout: 0.0s\n", "handoff_timeout"),
        ("handoff_timeout: -5\n", "handoff_timeout"),
        ("handoff_timeout: 5w\n", "handoff_timeout"),
        ("handoff_timeout: true\n", "handoff_timeout"),
        ("handoff_timeout: .inf\n", "handoff_timeout"),
        ("handoff_timeout: '9" + "9" * 400 + "'\n", "handoff_timeout"),
        ("auto_report_resume: 'yes'\n", "auto_report_resume"),
        ("auto_report_resume: 1\n", "auto_report_resume"),
        ("tombstone_days: 1.5\n", "tombstone_days"),
        ("tombstone_days: '3'\n", "tombstone_days"),
        ("tombstone_days: true\n", "tombstone_days"),
        ("prune: {older_than: 0}\n", "prune.older_than"),
        ("prune: {protect_labels: ['Bad Key']}\n", "prune.protect_labels.0"),
        ("prune: null\n", "prune"),
        ("kinds: {Bad: {}}\n", "kinds"),
        ("kinds: {x: {promote: maybe}}\n", "kinds.x.promote"),
        ("kinds: {x: {promote: null}}\n", "kinds.x.promote"),
        ("kinds: {x: {other: 1}}\n", "kinds.x.other"),
        ("repos: {nohost: {}}\n", "repos"),
        ("repos: {example.com/g/p: {kinds: {x: {promote: 1}}}}\n", "promote"),
        ("repos: {example.com/g/p: {extra: 1}}\n", "extra"),
        ("hooks: {post-launc: []}\n", "post-launc"),
        ("hooks: {post-launch: [{}]}\n", "hooks.post-launch.0.cmd"),
        ("hooks: {post-launch: [{cmd: ''}]}\n", "hooks.post-launch.0.cmd"),
        ('hooks: {post-launch: [{cmd: "a\\0b"}]}\n', "hooks.post-launch.0.cmd"),
        ("hooks: {post-launch: [{cmd: x, timeout: 0}]}\n", "hooks.post-launch.0.timeout"),
        ("hooks: {post-launch: [{cmd: x, states: [idle]}]}\n", "hooks.post-launch.0.states"),
        ("hooks: {pre-launch: [{cmd: x, async: true}]}\n", "hooks.pre-launch.0.async"),
        ("hooks: {pre-teardown: [{cmd: x, async: true}]}\n", "hooks.pre-teardown.0.async"),
        ("hooks: {post-launch: [{cmd: x, labels: [nokv]}]}\n", "hooks.post-launch.0.labels.0"),
        ("hooks: {post-launch: [{cmd: x, kinds: [BAD]}]}\n", "hooks.post-launch.0.kinds.0"),
        ("hooks: {post-launch: [{cmd: x, async: 'no'}]}\n", "hooks.post-launch.0.async"),
        (
            "hooks: {post-state-change: [{cmd: x, states: [nope]}]}\n",
            "hooks.post-state-change.0.states.0",
        ),
        (
            "hooks: {post-state-change: [{cmd: x, statuses: [nope]}]}\n",
            "hooks.post-state-change.0.statuses.0",
        ),
        ("hooks: {post-launch: {cmd: x}}\n", "hooks.post-launch"),
        # F1: unquoted huge numbers
        ("handoff_timeout: " + "9" * 400 + "\n", "handoff_timeout"),
        ("prune: {older_than: " + "9" * 400 + "}\n", "prune.older_than"),
        (
            "hooks: {post-launch: [{cmd: x, timeout: " + "9" * 400 + "}]}\n",
            "hooks.post-launch.0.timeout",
        ),
        # F5: YAML 1.2 core scalars; 1.1 forms are strings
        ("auto_report_resume: yes\n", "auto_report_resume"),
        ("auto_report_resume: on\n", "auto_report_resume"),
        ("hooks: {post-launch: [{cmd: x, async: no}]}\n", "hooks.post-launch.0.async"),
        ("handoff_timeout: 1:30\n", "handoff_timeout"),
        ("tombstone_days: 0x10\n", "tombstone_days"),
        ("tombstone_days: 0o10\n", "tombstone_days"),
        ("tombstone_days: 1_0\n", "tombstone_days"),
        ("tombstone_days: 10.0\n", "tombstone_days"),
        # F6: validate by alias only
        ("hooks: {pre_launch: [{cmd: x}]}\n", "hooks.pre_launch"),
        ("hooks: {post-launch: [{cmd: x, async_: true}]}\n", "hooks.post-launch.0.async_"),
        # F8: explicit null where the schema has no null
        ("repos: {example.com/g/p: {kinds: {x: {promote: null}}}}\n", "promote"),
        ("hooks: {post-launch: [{cmd: x, kinds: null}]}\n", "hooks.post-launch.0.kinds"),
        ("hooks: {post-launch: [{cmd: x, exclude_kinds: ~}]}\n", "exclude_kinds"),
        ("hooks: {post-launch: [{cmd: x, labels: null}]}\n", "hooks.post-launch.0.labels"),
        ("hooks: {post-state-change: [{cmd: x, states: null}]}\n", "states"),
        ("hooks: {post-state-change: [{cmd: x, statuses: null}]}\n", "statuses"),
    ],
)
def test_invalid_config_names_the_key(tmp_path: Path, text: str, key: str) -> None:
    assert key in invalid(tmp_path, text)


def test_protect_labels_error_has_a_plain_key_path(tmp_path: Path) -> None:
    message = invalid(tmp_path, "prune: {protect_labels: ['Bad Key']}\n")
    assert "key `prune.protect_labels.0`: not a valid label key or label" in message
    assert "function-after" not in message


def test_yaml_core_schema_scalars(tmp_path: Path) -> None:
    cfg = load(
        tmp_path,
        "tombstone_days: 010\nauto_report_resume: FALSE\nkinds: {no: {promote: True}, on: {}}\n"
        "me: {name: Null}\nhandoff_timeout: 1e2\n",
    )
    assert cfg.tombstone_days == 10  # decimal, not octal
    assert cfg.auto_report_resume is False
    assert set(cfg.kinds) == {"no", "on"}
    assert cfg.kinds["no"].promote is True
    assert cfg.me.name is None
    assert cfg.handoff_timeout == 100.0


@pytest.mark.parametrize(
    ("text", "path"),
    [
        ("123: x\n", "123"),
        ("kinds: {123: {}}\n", "kinds.123"),
        ("repos: {example.com/g/p: {kinds: {true: {}}}}\n", "repos.example.com/g/p.kinds.true"),
    ],
)
def test_non_string_key_names_the_key_path(tmp_path: Path, text: str, path: str) -> None:
    message = invalid(tmp_path, text)
    assert f"key `{path}`" in message
    assert "must be strings" in message


@pytest.mark.parametrize(
    "text",
    [
        "tombstone_days: !!int abc\n",
        "tombstone_days: !!int 3\n",
        "x: !!timestamp 2001-12-14\n",
        "auto_report_resume: !!bool maybe\n",
        "handoff_timeout: !!float abc\n",
        "me: !!map {name: a}\n",
        "claude_session_cmd: !!seq [a]\n",
        "me: {name: !!str a}\n",
        "me: {name: ! a}\n",
        "me: {name: !local a}\n",
    ],
)
def test_explicit_tags_are_refused_without_echo(tmp_path: Path, text: str) -> None:
    message = invalid(tmp_path, text)
    assert "line 1" in message
    assert "tags are not allowed" in message
    assert "abc" not in message and "maybe" not in message


def test_overlong_integer_is_config_invalid_without_echo(tmp_path: Path) -> None:
    message = invalid(tmp_path, "tombstone_days: " + "7" * 5000 + "\n")
    assert "7777" not in message


def test_valid_full_config(tmp_path: Path) -> None:
    cfg = load(
        tmp_path,
        """
me: {name: jo.e@x}
wtctl_cmd: null
claude_session_cmd: [uvx, claude-session]
prune: {older_than: 36h, protect_labels: [owner, 'pr=1']}
kinds: {implement: {promote: true}}
repos:
  example.com:8443/g/p: {kinds: {implement: {promote: false}}}
hooks:
  post-state-change:
    - {cmd: notify, async: true, timeout: '2m', states: [needs-input], statuses: [live],
       kinds: [implement], exclude_kinds: [review], labels: ['owner=me']}
  pre-launch: [{cmd: check, timeout: 0.5}]
""",
    )
    assert cfg.wtctl_cmd is None
    assert cfg.prune.older_than == 36 * 3600
    assert cfg.hooks.post_state_change[0].timeout == 120.0
    assert cfg.hooks.pre_launch[0].timeout == 0.5
    assert cfg.hooks.pre_launch[0].cmd == "check"
    assert cfg.hooks.post_launch == []


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("30s", 30.0), ("10m", 600.0), ("0.5s", 0.5), ("250ms", 0.25), ("2h", 7200.0),
     ("1d", 86400.0), ("90", 90.0), (90, 90.0), (1.5, 1.5)],
)  # fmt: skip
def test_duration_forms(tmp_path: Path, value: Any, seconds: float) -> None:
    cfg = load(tmp_path, f"handoff_timeout: {json.dumps(value)}\n")
    assert cfg.handoff_timeout == seconds


# --- promote resolution -----------------------------------------------------------------------


def _promote_cfg(tmp_path: Path) -> Config:
    return load(
        tmp_path,
        """
kinds: {a: {promote: true}, b: {promote: false}}
repos:
  example.com/g/p: {kinds: {a: {promote: false}, b: {promote: true}}}
  example.com/g/q: {kinds: {a: {}}}
""",
    )


def test_promote_flag_beats_repo_beats_kind_beats_default(tmp_path: Path) -> None:
    cfg = _promote_cfg(tmp_path)
    repo = "example.com/g/p"
    assert cfg.resolve_promote(kind="a", repo=None, explicit=None) is True  # kind
    assert cfg.resolve_promote(kind="a", repo=repo, explicit=None) is False  # repo over kind
    assert cfg.resolve_promote(kind="b", repo=repo, explicit=None) is True
    assert cfg.resolve_promote(kind="a", repo=repo, explicit=True) is True  # flag over repo
    assert cfg.resolve_promote(kind="b", repo=repo, explicit=False) is False
    assert (
        cfg.resolve_promote(kind="a", repo="example.com/g/q", explicit=None) is True
    )  # falls through
    assert cfg.resolve_promote(kind="zzz", repo=repo, explicit=None) is False  # default
    assert cfg.resolve_promote(kind=None, repo=None, explicit=None) is False


# --- hostile YAML -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "- a\n- b\n",
        "just a string\n",
        "42\n",
        "~ : 1\n",
        "1: x\n",
        "tombstone_days: 1\ntombstone_days: 2\n",
        "me: {name: a, name: b}\n",
        "x: &a [1]\ntombstone_days: *a\n",
        "a: &a [*a]\n",
        "tombstone_days: !!python/object/apply:os.system ['true']\n",
        "tombstone_days: !!binary Zm9v\n",
        "bad: [unclosed\n",
        "\t- tab\n",
        "k: \x01\n",
        "---\na: 1\n---\nb: 2\n",
    ],
)
def test_hostile_yaml_is_config_invalid(tmp_path: Path, text: str) -> None:
    invalid(tmp_path, text)


def test_billion_laughs_is_refused_quickly(tmp_path: Path) -> None:
    lines = ['a: &a ["x","x","x","x","x","x","x","x","x"]']
    prev = "a"
    for i in range(1, 10):
        name = chr(ord("a") + i)
        lines.append(f"{name}: &{name} [{', '.join(['*' + prev] * 9)}]")
        prev = name
    message = invalid(tmp_path, "\n".join(lines) + "\n")
    assert "alias" in message or "anchor" in message


def test_deep_nesting_is_config_invalid(tmp_path: Path) -> None:
    invalid(tmp_path, "a: " + "[" * 5000 + "]" * 5000 + "\n")


def test_oversized_file_is_refused(tmp_path: Path) -> None:
    invalid(tmp_path, "# " + "x" * (1024 * 1024 + 1) + "\ntombstone_days: 1\n")


def test_non_utf8_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "cfg" / "config.yaml"
    p.parent.mkdir()
    p.write_bytes(b"me: {name: \xff}\n")
    with pytest.raises(CliError):
        load_config(str(p), environ={})


def test_utf8_bom_and_unicode_comments_are_fine(tmp_path: Path) -> None:
    p = tmp_path / "cfg" / "config.yaml"
    p.parent.mkdir()
    p.write_bytes("﻿# héllo\ntombstone_days: 2\n".encode())
    assert load_config(str(p), environ={}).config.tombstone_days == 2


def test_directory_and_fifo_are_not_config_files(tmp_path: Path) -> None:
    (tmp_path / "d.yaml").mkdir()
    with pytest.raises(CliError):
        load_config(str(tmp_path / "d.yaml"), environ={})
    os.mkfifo(tmp_path / "f.yaml")
    with pytest.raises(CliError):
        load_config(str(tmp_path / "f.yaml"), environ={})


def test_unreadable_file_is_config_invalid(tmp_path: Path) -> None:
    p = write(tmp_path / "c.yaml", "tombstone_days: 1\n")
    p.chmod(0)
    try:
        if os.access(p, os.R_OK):
            pytest.skip("running as a user that ignores modes")
        with pytest.raises(CliError) as info:
            load_config(str(p), environ={})
        assert info.value.code == "config_invalid"
    finally:
        p.chmod(0o600)


def test_error_message_does_not_echo_secret_looking_values(tmp_path: Path) -> None:
    message = invalid(tmp_path, "tombstone_days: s3cr3t-token-value\n")
    assert "s3cr3t" not in message


def test_nul_in_path_is_config_invalid() -> None:
    with pytest.raises(CliError):
        load_config("/a\0b", environ={})
