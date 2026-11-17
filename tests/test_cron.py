"""``cron install`` and ``cron uninstall`` against the fake ``crontab`` (never the real one).

The table is the fake's file: foreign lines must survive every call byte for byte, in order.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import cron
from agent_spawn.output import CliError
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, agent_spawn_path

GOLDEN = Path(__file__).parent / "golden"
FOREIGN_TOP = "# my jobs\nMAILTO=me@example.invalid\n0 3 * * * /usr/bin/backup --all\n"
FOREIGN_BOTTOM = "\n@reboot /bin/true\n# last line\n"


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def table(fakebin: Fakebin) -> str | None:
    path = fakebin.root / "invocations.jsonl.crontab"
    return path.read_bytes().decode() if path.exists() else None


def put(fakebin: Fakebin, text: str) -> None:
    """Set the fake's table directly (not through a logged call)."""
    (fakebin.root / "invocations.jsonl.crontab").write_bytes(text.encode())


def writes(fakebin: Fakebin) -> list[list[str]]:
    """The crontab calls that change the table."""
    return [a for a in fakebin.argvs("crontab") if a != ["-l"]]


def run_json(env: Env, *args: str) -> tuple[int, Any]:
    done = env.run("--json", "cron", *args)
    return done.returncode, json.loads(done.stdout) if done.stdout.strip() else None


def block_of(text: str) -> list[str]:
    lines = text.split("\n")
    start = lines.index(cron.BEGIN)
    return lines[start : lines.index(cron.END) + 1]


def normalized(env: Env, body: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(body)
    for real, name in (
        (agent_spawn_path(), "<AGENT_SPAWN>"),
        (os.path.dirname(agent_spawn_path()), "<BIN>"),
        (str(env.iso.home), "<HOME>"),
        (str(env.iso.state), "<STATE>"),
        (str(env.iso.config), "<CONFIG>"),
    ):
        text = text.replace(json.dumps(real)[1:-1], name)
    return json.loads(text)


# --- install -----------------------------------------------------------------------------------


def test_install_into_no_crontab_writes_exactly_the_block(env: Env, fakebin: Fakebin) -> None:
    code, body = run_json(env, "install")
    assert code == 0
    schema_registry.validate(body, "agent-spawn/cli/cron-install.json")
    assert body["action"] == "installed" and body["interval_minutes"] == 10
    assert body["dry_run"] is False and body["env_file"] is None
    assert table(fakebin) == body["block"] + "\n"
    assert fakebin.argvs("crontab") == [["-l"], ["-"]]
    lines = body["block"].split("\n")
    assert lines[0] == "# agent-spawn managed" and lines[2] == "# agent-spawn managed end"
    job = lines[1]
    assert job.startswith("*/10 * * * * ")
    assert job.endswith(f"'{agent_spawn_path()}' reconcile --quiet")
    path = job.split("PATH='")[1].split("'")[0].split(":")
    assert str(env.iso.home / ".local" / "bin") in path
    assert os.path.dirname(agent_spawn_path()) in path
    assert f"XDG_STATE_HOME='{env.iso.state}'" in job


def test_golden_install_and_uninstall_bodies(env: Env, fakebin: Fakebin) -> None:
    _, installed = run_json(env, "install", "--interval", "5")
    _, removed = run_json(env, "uninstall")
    for name, body in (("cron_install", installed), ("cron_uninstall", removed)):
        golden = json.loads((GOLDEN / f"{name}.json").read_text())
        assert normalized(env, body) == golden


def test_install_replaces_the_block_in_place_and_keeps_foreign_lines(
    env: Env, fakebin: Fakebin
) -> None:
    run_json(env, "install", "--interval", "7")
    mine = table(fakebin)
    assert mine is not None
    put(fakebin, FOREIGN_TOP + mine + FOREIGN_BOTTOM)
    code, body = run_json(env, "install", "--interval", "20")
    assert code == 0 and body["action"] == "updated" and body["interval_minutes"] == 20
    after = table(fakebin)
    assert after is not None
    assert after.startswith(FOREIGN_TOP + cron.BEGIN + "\n*/20 * * * * ")
    assert after.endswith(cron.END + "\n" + FOREIGN_BOTTOM)
    assert after.split("\n").count(cron.BEGIN) == 1


def test_install_appends_after_foreign_lines(env: Env, fakebin: Fakebin) -> None:
    put(fakebin, FOREIGN_TOP + FOREIGN_BOTTOM)
    _, body = run_json(env, "install")
    assert body["action"] == "installed"
    assert table(fakebin) == FOREIGN_TOP + FOREIGN_BOTTOM + body["block"] + "\n"


def test_installing_the_same_block_again_writes_nothing(env: Env, fakebin: Fakebin) -> None:
    run_json(env, "install")
    before = table(fakebin)
    _, body = run_json(env, "install")
    assert body["action"] == "unchanged" and table(fakebin) == before
    assert writes(fakebin) == [["-"]]  # only the first install wrote


def test_install_collapses_duplicate_blocks_into_one(env: Env, fakebin: Fakebin) -> None:
    run_json(env, "install")
    mine = table(fakebin) or ""
    put(fakebin, mine + "# between\n" + mine + FOREIGN_BOTTOM)
    _, body = run_json(env, "install")
    assert body["action"] == "updated"
    assert table(fakebin) == mine + "# between\n" + FOREIGN_BOTTOM


@pytest.mark.parametrize("interval", ["1", "59"])
def test_interval_bounds_are_accepted(env: Env, interval: str) -> None:
    _, body = run_json(env, "install", "--interval", interval)
    assert body["interval_minutes"] == int(interval)
    assert f"*/{interval} * * * * " in body["block"]


@pytest.mark.parametrize("interval", ["0", "60", "-1", "x", "10.5", ""])
def test_a_bad_interval_is_exit_2_and_touches_nothing(
    env: Env, fakebin: Fakebin, interval: str
) -> None:
    done = env.run("--json", "cron", "install", "--interval", interval)
    assert done.returncode == 2
    assert fakebin.argvs("crontab") == []


def test_foreign_unicode_line_breaks_and_crlf_survive(env: Env, fakebin: Fakebin) -> None:
    foreign = "# a b\x0bc\x85d\r\n* * * * * echo é中\r\n"
    put(fakebin, foreign)
    run_json(env, "install")
    assert (table(fakebin) or "").startswith(foreign + cron.BEGIN + "\n")
    run_json(env, "uninstall")
    assert table(fakebin) == foreign


def test_the_env_file_is_named_not_read(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    secret = "TOKEN=hunter2-never-in-the-crontab"
    envfile = tmp_path / "agent.env"
    envfile.write_text(secret + "\n")
    envfile.chmod(0o600)
    done = env.run("--json", "cron", "install", "--env-file", str(envfile))
    assert done.returncode == 0
    body = json.loads(done.stdout)
    schema_registry.validate(body, "agent-spawn/cli/cron-install.json")
    assert body["env_file"] == str(envfile)
    job = body["block"].split("\n")[1]
    assert f"set -a && . '{envfile}' && set +a && " in job
    for text in (table(fakebin) or "", done.stdout, done.stderr):
        assert "hunter2" not in text


@pytest.mark.parametrize(
    "case",
    ["relative", "missing", "directory", "group_writable", "world_writable", "newline"],
)
def test_a_bad_env_file_is_exit_2_before_the_crontab_is_touched(
    env: Env, fakebin: Fakebin, tmp_path: Path, case: str
) -> None:
    path = tmp_path / "e.env"
    path.write_text("A=1\n")
    path.chmod(0o600)
    arg = str(path)
    if case == "relative":
        arg = "e.env"
    elif case == "missing":
        arg = str(tmp_path / "nope")
    elif case == "directory":
        arg = str(tmp_path)
    elif case == "group_writable":
        path.chmod(0o620)
    elif case == "world_writable":
        path.chmod(0o602)
    elif case == "newline":
        arg = str(tmp_path / "a\nb")
    done = env.run("--json", "cron", "install", "--env-file", arg)
    assert done.returncode == 2, done.stderr
    assert json.loads(done.stdout)["error"]["code"] == "usage"
    assert fakebin.argvs("crontab") == []


def test_an_env_file_owned_by_someone_else_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "e.env"
    path.write_text("A=1\n")
    path.chmod(0o600)
    assert cron.check_env_file(str(path), os.getuid()) == str(path)
    with pytest.raises(CliError, match="not owned"):
        cron.check_env_file(str(path), os.getuid() + 1)


HOSTILE = [
    "plain",
    "it's",
    "a b",
    "100%",
    "$(touch pwned)",
    "`touch pwned`",
    "a;b|c&d",
    "back\\slash",
    '"dq"',
    "~/x",
    "*",
    "-n",
    "é中",
]


@pytest.mark.parametrize("value", HOSTILE)
def test_quoting_round_trips_through_sh_and_cron_percent_rules(value: str, tmp_path: Path) -> None:
    line = cron.cron_escape(f"printf %s {cron.shell_quote(value)}")
    assert "%" not in line.replace("\\%", "")  # every % is escaped
    # cron removes the backslash of an escaped % before the shell sees the command
    command = line.replace("\\%", "%")
    done = subprocess.run(
        ["sh", "-c", command], capture_output=True, text=True, cwd=tmp_path, check=True
    )
    assert done.stdout == value
    assert not (tmp_path / "pwned").exists()


def test_a_hostile_env_file_path_stays_one_quoted_word(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    folder = tmp_path / "it's 100% $(touch pwned);x"
    folder.mkdir()
    envfile = folder / "e.env"
    envfile.write_text("A=1\n")
    envfile.chmod(0o600)
    done = env.run("--json", "cron", "install", "--env-file", str(envfile))
    assert done.returncode == 0, done.stderr
    job = json.loads(done.stdout)["block"].split("\n")[1]
    assert "\\%" in job and "%" not in job.replace("\\%", "")
    assert " '" + str(envfile).replace("'", "'\\''").replace("%", "\\%") + "' " in job
    assert "\n" not in job
    assert (table(fakebin) or "").count("\n") == 3


def test_an_explicit_config_is_passed_with_an_absolute_path(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    config = tmp_path / "my config.yaml"
    config.write_text("{}\n")
    done = env.run("--json", "--config", str(config), "cron", "install")
    assert done.returncode == 0, done.stderr
    job = json.loads(done.stdout)["block"].split("\n")[1]
    assert f"'{agent_spawn_path()}' --config '{config}' reconcile --quiet" in job


def test_a_config_path_with_a_control_character_fails_before_any_write(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    config = tmp_path / "a\nb.yaml"
    config.write_text("{}\n")
    done = env.run("--json", "--config", str(config), "cron", "install")
    assert done.returncode == 12
    assert json.loads(done.stdout)["error"]["code"] == "failed"
    assert fakebin.argvs("crontab") == []


def test_too_long_a_line_is_refused(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    deep = tmp_path.joinpath(*["d" * 60] * 16)
    deep.mkdir(parents=True)
    envfile = deep / "e.env"
    envfile.write_text("A=1\n")
    envfile.chmod(0o600)
    done = env.run("--json", "cron", "install", "--env-file", str(envfile))
    assert done.returncode == 12 and "longer than" in json.loads(done.stdout)["error"]["message"]
    assert fakebin.argvs("crontab") == []


# --- dry run -----------------------------------------------------------------------------------


def test_install_dry_run_reports_and_never_writes(env: Env, fakebin: Fakebin) -> None:
    put(fakebin, FOREIGN_TOP)
    code, body = run_json(env, "install", "--dry-run")
    assert code == 0 and body["dry_run"] is True and body["action"] == "installed"
    schema_registry.validate(body, "agent-spawn/cli/cron-install.json")
    assert table(fakebin) == FOREIGN_TOP
    assert ["-"] not in fakebin.argvs("crontab")[1:]
    # the dry run of an installed block says what a real run would do
    run_json(env, "install")
    code, body = run_json(env, "install", "--dry-run", "--interval", "3")
    assert body["action"] == "updated"
    _, body = run_json(env, "install", "--dry-run")
    assert body["action"] == "unchanged"


def test_uninstall_dry_run_reports_and_never_writes(env: Env, fakebin: Fakebin) -> None:
    run_json(env, "install")
    before = table(fakebin)
    calls = len(fakebin.argvs("crontab"))
    code, body = run_json(env, "uninstall", "--dry-run")
    assert code == 0 and body["dry_run"] is True and body["action"] == "removed"
    schema_registry.validate(body, "agent-spawn/cli/cron-uninstall.json")
    assert table(fakebin) == before
    assert fakebin.argvs("crontab")[calls:] == [["-l"]]


# --- uninstall ---------------------------------------------------------------------------------


def test_uninstall_removes_only_the_block(env: Env, fakebin: Fakebin) -> None:
    run_json(env, "install")
    put(fakebin, FOREIGN_TOP + (table(fakebin) or "") + FOREIGN_BOTTOM)
    code, body = run_json(env, "uninstall")
    assert code == 0 and body["action"] == "removed"
    schema_registry.validate(body, "agent-spawn/cli/cron-uninstall.json")
    assert body["block"].split("\n")[0] == cron.BEGIN
    assert table(fakebin) == FOREIGN_TOP + FOREIGN_BOTTOM


def test_uninstall_without_a_block_is_a_no_op_exit_0(env: Env, fakebin: Fakebin) -> None:
    put(fakebin, FOREIGN_TOP)
    code, body = run_json(env, "uninstall")
    assert (code, body) == (0, {"action": "unchanged", "block": None, "dry_run": False})
    assert table(fakebin) == FOREIGN_TOP
    assert writes(fakebin) == []


def test_uninstall_without_any_crontab_is_a_no_op_exit_0(env: Env, fakebin: Fakebin) -> None:
    code, body = run_json(env, "uninstall")
    assert code == 0 and body["action"] == "unchanged"
    assert fakebin.argvs("crontab") == [["-l"]]


def test_uninstalling_the_only_content_removes_the_crontab(env: Env, fakebin: Fakebin) -> None:
    run_json(env, "install")
    code, body = run_json(env, "uninstall")
    assert code == 0 and body["action"] == "removed"
    assert table(fakebin) is None
    assert fakebin.argvs("crontab")[-1] == ["-r"]


def test_install_then_uninstall_restores_the_foreign_table(env: Env, fakebin: Fakebin) -> None:
    original = FOREIGN_TOP + FOREIGN_BOTTOM
    put(fakebin, original)
    run_json(env, "install")
    run_json(env, "uninstall")
    assert table(fakebin) == original


# --- failures never write ----------------------------------------------------------------------


@pytest.mark.parametrize("command", ["install", "uninstall"])
def test_a_failing_read_aborts_without_writing(env: Env, fakebin: Fakebin, command: str) -> None:
    fakebin.script("crontab", "list", exit=1, stderr="crontab: permission denied\n")
    done = env.run("--json", "cron", command)
    assert done.returncode == 12
    error = json.loads(done.stdout)["error"]
    assert error["code"] == "failed" and "permission denied" in error["message"]
    assert fakebin.argvs("crontab") == [["-l"]]


@pytest.mark.parametrize("command", ["install", "uninstall"])
def test_a_read_with_exit_2_or_partial_output_is_not_an_empty_table(
    env: Env, fakebin: Fakebin, command: str
) -> None:
    fakebin.script("crontab", "list", exit=2, stdout=FOREIGN_TOP, stderr="no crontab for me\n")
    assert env.run("--json", "cron", command).returncode == 12
    assert fakebin.argvs("crontab") == [["-l"]]


def test_a_failing_write_is_exit_12(env: Env, fakebin: Fakebin) -> None:
    fakebin.script("crontab", "install", exit=1, stderr="crontab: no space left\n")
    done = env.run("--json", "cron", "install")
    assert done.returncode == 12
    assert "no space left" in json.loads(done.stdout)["error"]["message"]


def test_no_crontab_command_is_exit_12(env: Env, tmp_path: Path) -> None:
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    for command in ("install", "uninstall"):
        done = env.run("--json", "cron", command, extra_env={"PATH": str(empty)})
        assert done.returncode == 12, done.stderr
        error = json.loads(done.stdout)["error"]
        assert error["code"] == "failed" and "crontab" in error["message"]


def test_an_unterminated_block_is_refused_and_nothing_is_written(
    env: Env, fakebin: Fakebin
) -> None:
    broken = (
        FOREIGN_TOP
        + cron.BEGIN
        + "\n*/5 * * * * agent-spawn reconcile --quiet\n"
        + "0 1 * * * /x\n"
    )
    put(fakebin, broken)
    for command in ("install", "uninstall"):
        done = env.run("--json", "cron", command)
        assert done.returncode == 12
        assert "no `# agent-spawn managed end`" in json.loads(done.stdout)["error"]["message"]
    assert table(fakebin) == broken
    assert writes(fakebin) == []


def test_a_foreign_line_inside_the_block_is_refused_not_swallowed(
    env: Env, fakebin: Fakebin
) -> None:
    broken = f"{cron.BEGIN}\n*/5 * * * * agent-spawn reconcile --quiet\n0 1 * * * /x\n{cron.END}\n"
    put(fakebin, broken)
    for command in ("install", "uninstall"):
        done = env.run("--json", "cron", command)
        assert done.returncode == 12
        assert "line 3" in json.loads(done.stdout)["error"]["message"]
    assert table(fakebin) == broken


def test_a_stray_end_marker_alone_is_a_foreign_comment(env: Env, fakebin: Fakebin) -> None:
    put(fakebin, cron.END + "\n")
    _, body = run_json(env, "install")
    assert body["action"] == "installed"
    assert (table(fakebin) or "").startswith(cron.END + "\n" + cron.BEGIN)


# --- the pure parts ----------------------------------------------------------------------------


def test_installed_and_uninstalled_on_plain_lines() -> None:
    block = [cron.BEGIN, "*/10 * * * * x reconcile --quiet", cron.END]
    action, out = cron.installed(["a", "b"], block)
    assert (action, out) == ("installed", ["a", "b", *block])
    assert cron.uninstalled(out) == (["a", "b"], block)
    assert cron.installed(out, block)[0] == "unchanged"
    assert cron.split_lines("") == [] and cron.split_lines("a\n\n") == ["a", ""]


def test_the_leak_guard_refuses_the_real_crontab(
    isolation_env: Isolation, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_spawn import gateway

    real = "/usr/bin/crontab"
    if not os.path.exists(real):
        pytest.skip("no real crontab to guard against")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    with pytest.raises(AssertionError, match="real crontab"):
        gateway.SystemGateway().run(["crontab", "-l"])
