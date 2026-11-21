"""``validate``: offline, read-only, one verdict for the harness's ``doctor``."""

from __future__ import annotations

import hashlib
import json
import os
import pty
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import checking, cli
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env

GOLDEN = Path(__file__).parent / "golden"
SCHEMA = "agent-spawn/cli/validate.json"
TOOLS = ("claude-session", "termctl", "wtctl")


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def run_json(env: Env, *args: str, extra_env: dict[str, str] | None = None) -> tuple[int, Any, str]:
    done = env.run("--json", "validate", *args, extra_env=extra_env)
    body = json.loads(done.stdout) if done.stdout.strip() else None
    return done.returncode, body, done.stderr


def script(tmp: Path, name: str, body: str) -> str:
    path = tmp / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


def snapshot(*roots: Path) -> dict[str, tuple[Any, ...]]:
    """Every entry under ``roots`` with mode, size, mtime and a content hash."""
    seen: dict[str, tuple[Any, ...]] = {}
    for root in roots:
        seen[str(root)] = (stat.S_IMODE(root.lstat().st_mode), root.lstat().st_mtime_ns)
        for path in sorted(root.rglob("*")):
            info = path.lstat()
            digest = (
                hashlib.sha256(path.read_bytes()).hexdigest() if stat.S_ISREG(info.st_mode) else ""
            )
            seen[str(path)] = (info.st_mode, info.st_size, info.st_mtime_ns, digest)
    return seen


# --- verdicts ----------------------------------------------------------------------------------


def test_valid_defaults_exit_0_with_no_warnings(env: Env, fakebin: Fakebin) -> None:
    code, body, err = run_json(env)
    assert code == 0 and err == ""
    schema_registry.validate(body, SCHEMA)
    assert body["warnings"] == [] and body["probe"] is False
    assert body["config"] == {"path": str(env.config), "files": []}
    assert body["checks"]["state_dir"] == {"path": str(env.state_dir), "status": "absent"}
    for name in TOOLS:
        tool = body["checks"]["tools"][name]
        assert tool["status"] == "ok" and tool["version"] is None
        assert tool["path"] == str(fakebin.bin_dir / name)


def test_golden_body(env: Env, fakebin: Fakebin) -> None:
    _, body, _ = run_json(env)
    text = json.dumps(body)
    for real, name in (
        (str(fakebin.bin_dir), "<BIN>"),
        (str(env.state_dir), "<STATE>/agent-spawn"),
        (str(env.iso.config), "<CONFIG>"),
    ):
        text = text.replace(real, name)
    assert json.loads(text) == json.loads((GOLDEN / "validate.json").read_text())


def test_config_and_dropins_are_listed_and_state_dir_ok(env: Env) -> None:
    env.config.parent.mkdir(parents=True)
    env.config.write_text("tombstone_days: 3\n")
    drop = env.config.parent / "config.d"
    drop.mkdir()
    (drop / "b.yaml").write_text("tombstone_days: 4\n")
    (drop / "a.yaml").write_text("me: {name: x}\n")
    env.state_dir.mkdir(parents=True)
    code, body, _ = run_json(env)
    assert code == 0
    assert body["config"]["files"] == [str(env.config), str(drop / "a.yaml"), str(drop / "b.yaml")]
    assert body["checks"]["state_dir"]["status"] == "ok"


def test_invalid_key_exits_2_naming_it(env: Env) -> None:
    env.config.parent.mkdir(parents=True)
    env.config.write_text("tombstone_days: 3\nnot_a_key: 1\n")
    code, body, err = run_json(env)
    assert code == 2
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["code"] == "config_invalid" and "not_a_key" in body["error"]["message"]
    assert "not_a_key" in err


def test_invalid_config_without_json_prints_nothing_on_stdout(env: Env) -> None:
    env.config.parent.mkdir(parents=True)
    env.config.write_text("tombstone_days: -1\n")
    done = env.run("validate")
    assert done.returncode == 2 and done.stdout == "" and "tombstone_days" in done.stderr


@pytest.mark.parametrize(
    "data",
    [b"me: {name: \xff}\n", b"a: " + b"x" * (1024 * 1024 + 10), b"\x00\x01"],
    ids=["non-utf8", "oversized", "binary"],
)
def test_hostile_config_file_is_config_invalid(env: Env, data: bytes) -> None:
    env.config.parent.mkdir(parents=True)
    env.config.write_bytes(data)
    code, body, _ = run_json(env)
    assert code == 2 and body["error"]["code"] == "config_invalid"


def test_config_that_is_a_directory_or_missing_named_file(env: Env, tmp_path: Path) -> None:
    code, body, _ = run_json(env, "--config", str(tmp_path / "gone.yaml"))
    assert code == 2 and body["error"]["code"] == "config_invalid"
    code, body, _ = run_json(env, "--config", str(tmp_path))
    assert code == 2 and body["error"]["code"] == "config_invalid"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads everything")
def test_unreadable_config_is_config_invalid(env: Env) -> None:
    env.config.parent.mkdir(parents=True)
    env.config.write_text("tombstone_days: 3\n")
    env.config.chmod(0)
    code, body, _ = run_json(env)
    assert code == 2 and body["error"]["code"] == "config_invalid"


def test_named_config_and_state_home_are_honored(env: Env, tmp_path: Path) -> None:
    named = tmp_path / "other.yaml"
    named.write_text("tombstone_days: 3\n")
    other_state = tmp_path / "st"
    code, body, _ = run_json(
        env, "--config", str(named), extra_env={"XDG_STATE_HOME": str(other_state)}
    )
    assert code == 0
    assert body["config"] == {"path": str(named), "files": [str(named)]}
    assert body["checks"]["state_dir"]["path"] == str(other_state / "agent-spawn")
    assert not other_state.exists()


# --- tools -------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["claude-session", "wtctl"])
def test_missing_required_tool_is_tool_missing_exit_12(env: Env, name: str, tmp_path: Path) -> None:
    key = {"claude-session": "claude_session_cmd", "wtctl": "wtctl_cmd"}[name]
    env.config.parent.mkdir(parents=True)
    env.config.write_text(f"{key}: [{tmp_path}/nope]\n")
    code, body, err = run_json(env)
    assert code == 12
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["code"] == "tool_missing" and body["error"]["tool"] == name
    assert name in err


def test_both_missing_names_both_and_reports_claude_session_first(env: Env, tmp_path: Path) -> None:
    env.config.parent.mkdir(parents=True)
    env.config.write_text(f"claude_session_cmd: [{tmp_path}/a]\nwtctl_cmd: [{tmp_path}/b]\n")
    code, body, _ = run_json(env)
    assert code == 12 and body["error"]["tool"] == "claude-session"
    assert "claude-session" in body["error"]["message"] and "wtctl" in body["error"]["message"]


def test_a_non_executable_tool_is_missing(env: Env, tmp_path: Path) -> None:
    plain = tmp_path / "claude-session"
    plain.write_text("#!/bin/sh\n")
    env.config.parent.mkdir(parents=True)
    env.config.write_text(f"claude_session_cmd: [{plain}]\n")
    code, body, _ = run_json(env)
    assert code == 12 and body["error"]["code"] == "tool_missing"


def test_wtctl_null_is_disabled_without_a_warning(env: Env) -> None:
    env.config.parent.mkdir(parents=True)
    env.config.write_text("wtctl_cmd: null\n")
    code, body, err = run_json(env)
    assert code == 0 and err == ""
    assert body["checks"]["tools"]["wtctl"] == {
        "status": "disabled",
        "path": None,
        "version": None,
        "minimum": "0.1.0",
    }
    assert body["warnings"] == []


def test_missing_termctl_is_a_warning(env: Env, tmp_path: Path) -> None:
    env.config.parent.mkdir(parents=True)
    env.config.write_text(f"termctl_cmd: [{tmp_path}/gone]\n")
    code, body, err = run_json(env)
    assert code == 0
    schema_registry.validate(body, SCHEMA)
    assert [w["code"] for w in body["warnings"]] == ["termctl_missing"]
    assert body["warnings"][0]["key"] == "termctl_cmd"
    assert body["checks"]["tools"]["termctl"]["status"] == "missing"
    assert err.startswith("agent-spawn: warning: termctl_cmd: ") and err.count("\n") == 1
    done = env.run("--json", "validate", "--quiet")
    assert done.stderr == "" and json.loads(done.stdout)["warnings"]


def test_scope_without_systemd_run_is_a_warning(env: Env, tmp_path: Path) -> None:
    env.config.parent.mkdir(parents=True)
    env.config.write_text("supervisor: {detach: scope}\n")
    code, body, _ = run_json(env)  # the fakebin has a systemd-run
    assert code == 0 and body["warnings"] == []
    bare = tmp_path / "bare"
    bare.mkdir()
    for name in (*TOOLS, "python3"):
        (bare / name).symlink_to(env.fakebin.bin_dir / name)
    code, body, _ = run_json(env, extra_env={"PATH": str(bare)})
    assert code == 0, body
    assert [w["code"] for w in body["warnings"]] == ["systemd_run_missing"]


def test_a_tool_path_with_hostile_text_is_escaped(env: Env) -> None:
    env.config.parent.mkdir(parents=True)
    env.config.write_text('termctl_cmd: ["/no/\\u202e\\x1b[31m' + "x" * 500 + '"]\n')
    done = env.run("validate")
    assert done.returncode == 0
    assert "\x1b" not in done.stderr and "‮" not in done.stderr
    assert len(done.stderr) < 600


# --- read-only ---------------------------------------------------------------------------------


@pytest.mark.parametrize("args", [[], ["--probe"]])
@pytest.mark.parametrize("prepared", [False, True])
def test_the_fresh_tree_is_byte_identical_after(env: Env, args: list[str], prepared: bool) -> None:
    if prepared:
        env.config.parent.mkdir(parents=True)
        env.config.write_text("tombstone_days: 3\n")
        env.state_dir.mkdir(parents=True)
    roots = [env.iso.home, env.iso.config, env.iso.state, env.iso.data, env.iso.cache]
    before = snapshot(*roots)
    code, _, _ = run_json(env, *args)
    assert code == 0
    assert snapshot(*roots) == before
    assert env.state_dir.exists() is prepared


def test_failing_validate_also_changes_nothing(env: Env, tmp_path: Path) -> None:
    env.config.parent.mkdir(parents=True)
    env.config.write_text(f"wtctl_cmd: [{tmp_path}/x]\n")
    roots = [env.iso.home, env.iso.config, env.iso.state]
    before = snapshot(*roots)
    assert run_json(env, "--probe")[0] == 12
    assert snapshot(*roots) == before


def test_state_dir_that_is_a_file_fails(env: Env) -> None:
    env.iso.state.mkdir(exist_ok=True)
    env.state_dir.write_text("x")
    code, body, _ = run_json(env)
    assert code == 12 and body["error"]["code"] == "failed"
    assert body["error"]["message"].startswith("state_dir:")


def test_state_dir_below_a_file_or_dangling_link_fails(env: Env, tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    code, body, _ = run_json(env, extra_env={"XDG_STATE_HOME": str(blocker / "sub")})
    assert code == 12 and body["error"]["message"].startswith("state_dir:")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "nowhere")
    code, body, _ = run_json(env, extra_env={"XDG_STATE_HOME": str(link)})
    assert code == 12 and body["error"]["message"].startswith("state_dir:")


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes everywhere")
def test_unwritable_state_dir_fails(env: Env) -> None:
    env.state_dir.mkdir(parents=True)
    env.state_dir.chmod(0o500)
    try:
        code, body, _ = run_json(env)
    finally:
        env.state_dir.chmod(0o700)
    assert code == 12 and body["error"]["message"].startswith("state_dir:")


# --- --probe -----------------------------------------------------------------------------------


def test_plain_validate_starts_no_process_and_probe_runs_each_tool_once(
    env: Env, fakebin: Fakebin
) -> None:
    run_json(env)
    assert fakebin.argvs("claude-session") == [] and fakebin.argvs("wtctl") == []
    code, body, _ = run_json(env, "--probe", extra_env={"FAKEBIN_VERSION": "0.43.0"})
    assert code == 0 and body["probe"] is True and body["warnings"] == []
    schema_registry.validate(body, SCHEMA)
    for name in TOOLS:
        assert body["checks"]["tools"][name]["version"] == "0.43.0"
        assert fakebin.argvs(name) == [["--version"]]


def test_probe_warns_when_claude_session_cannot_do_implicit_done(env: Env) -> None:
    code, body, err = run_json(env, "--probe")  # the fakes print 0.1.0
    assert code == 0
    assert [(w["code"], w["key"]) for w in body["warnings"]] == [
        ("implicit_done_unavailable", "claude_session_cmd")
    ]
    assert "0.43.0" in err


def test_probe_flags_a_version_below_the_minimum(env: Env) -> None:
    code, body, _ = run_json(env, "--probe", extra_env={"FAKEBIN_VERSION": "0.0.9"})
    assert code == 0
    assert {w["code"] for w in body["warnings"]} == {"tool_too_old"}
    assert len(body["warnings"]) == 3


@pytest.mark.parametrize(
    "body",
    [
        "exit 3",
        "echo garbage",
        "echo 'claude-session 1.2'",
        "echo 'claude-session 1.2.3-rc1'",
        "echo 'wtctl 1.2.3'",
        "echo 'claude-session 1.2.3 extra'",
        "echo 'claude-session 0.0.0'",
        "printf '\\377\\376 \\000'",
        "",
        "echo 'claude-session 1.2.3' >&2; exit 1",
    ],
)
def test_probe_trouble_is_a_warning_never_a_failure(env: Env, tmp_path: Path, body: str) -> None:
    tool = script(tmp_path, "cs", body + "\n")
    env.config.parent.mkdir(parents=True)
    env.config.write_text(f"claude_session_cmd: [{tool}]\n")
    code, out, _ = run_json(env, "--probe", extra_env={"FAKEBIN_VERSION": "0.43.0"})
    assert code == 0
    schema_registry.validate(out, SCHEMA)
    assert [w["code"] for w in out["warnings"]] == ["tool_version_unknown"]
    assert out["checks"]["tools"]["claude-session"]["version"] is None


def test_probe_accepts_a_good_version_from_a_script(env: Env, tmp_path: Path) -> None:
    tool = script(tmp_path, "cs", 'echo "claude-session 1.2.3"\n')
    env.config.parent.mkdir(parents=True)
    env.config.write_text(f"claude_session_cmd: [{tool}]\n")
    code, out, _ = run_json(env, "--probe", extra_env={"FAKEBIN_VERSION": "0.43.0"})
    assert code == 0 and out["warnings"] == []
    assert out["checks"]["tools"]["claude-session"]["version"] == "1.2.3"


def test_probe_passes_the_extra_argv_then_version(env: Env, tmp_path: Path) -> None:
    mark = tmp_path / "args"
    tool = script(tmp_path, "cs", f'echo "$@" > {mark}\necho "claude-session 1.0.0"\n')
    env.config.parent.mkdir(parents=True)
    env.config.write_text(f"claude_session_cmd: [{tool}, run, --x]\n")
    assert run_json(env, "--probe")[0] == 0
    assert mark.read_text() == "run --x --version\n"


def test_a_hanging_tool_times_out_into_a_warning(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pidfile = tmp_path / "pid"
    tool = script(tmp_path, "cs", f"echo $$ > {pidfile}\nexec sleep 30\n")
    env.config.parent.mkdir(parents=True)
    env.config.write_text(f"claude_session_cmd: [{tool}]\n")
    monkeypatch.setattr(checking, "PROBE_TIMEOUT", 0.5)
    with monkeypatch.context() as scoped:  # gone before the isolation fixture's teardown check
        scoped.setenv("PATH", f"{env.fakebin.bin_dir}:{os.environ['PATH']}")
        scoped.setenv("FAKEBIN_LOG", str(env.fakebin.log))
        scoped.setenv("FAKEBIN_VERSION", "0.43.0")
        assert cli.main(["--json", "validate", "--probe"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [w["code"] for w in out["warnings"]] == ["tool_version_unknown"]
    assert "within" in out["warnings"][0]["message"]
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):  # the exact child was killed and reaped
        os.kill(pid, 0)


def test_probe_that_cannot_start_the_program_is_a_warning(env: Env, tmp_path: Path) -> None:
    bad = tmp_path / "cs"
    bad.write_text("not a script, no shebang\n")
    bad.chmod(0o755)
    env.config.parent.mkdir(parents=True)
    env.config.write_text(f"claude_session_cmd: [{bad}]\n")
    code, out, _ = run_json(env, "--probe")
    assert code == 0 and "tool_version_unknown" in {w["code"] for w in out["warnings"]}


# --- usage, streams and prompts ------------------------------------------------------------------


@pytest.mark.parametrize("extra", [["--dry-run"], ["extra"], ["--prob"]])
def test_unknown_flags_and_positionals_are_usage_errors(env: Env, extra: list[str]) -> None:
    code, body, _ = run_json(env, *extra)
    assert code == 2 and body["error"]["code"] == "usage"


def test_json_is_one_line_and_human_output_is_a_summary(env: Env) -> None:
    done = env.run("--json", "validate")
    assert done.stdout.count("\n") == 1
    human = env.run("validate")
    assert human.returncode == 0 and human.stderr == ""
    assert "claude-session:" in human.stdout and human.stdout.rstrip().endswith("ok")


def test_flags_may_precede_or_follow(env: Env) -> None:
    assert env.run("validate", "--json").stdout == env.run("--json", "validate").stdout


def test_never_prompts_even_on_a_terminal(env: Env) -> None:
    master, slave = pty.openpty()
    try:
        done = subprocess.run(
            env.argv("validate", "--probe"),
            stdin=slave,
            capture_output=True,
            text=True,
            env=env.env(),
            timeout=30,
            check=False,
        )
    finally:
        os.close(master)
        os.close(slave)
    assert done.returncode == 0


def test_schema_rejects_what_it_should() -> None:
    good = {
        "probe": False,
        "config": {"path": "/c", "files": []},
        "checks": {
            "state_dir": {"path": "/s", "status": "ok"},
            "tools": {
                n: {"status": "ok", "path": "/b", "version": None, "minimum": "0.1.0"}
                for n in TOOLS
            },
        },
        "warnings": [],
    }
    schema_registry.validate(good, SCHEMA)
    for bad in (
        {**good, "extra": 1},
        {k: v for k, v in good.items() if k != "warnings"},
        {**good, "warnings": [{"code": "nope", "message": "m"}]},
        {**good, "warnings": ["a string"]},
    ):
        assert list(schema_registry.validator(SCHEMA).iter_errors(bad))
