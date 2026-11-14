"""The fake termctl: canned outputs validate against the pinned schemas, argv is checked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import sibling_schemas
from tests.fakebin.harness import Fakebin
from tests.fakebin_support import error_of, run_json


def open_pane(fakebin: Fakebin, *extra: str, cwd: Path | None = None) -> dict[str, object]:
    code, body = run_json(
        fakebin, "termctl", "open", *extra, "--", "sleep", "30", schema="cli/open.json", cwd=cwd
    )
    assert code == 0
    return body


def test_open_validates_and_records_argv_verbatim_after_the_separator(fakebin: Fakebin) -> None:
    pane = open_pane(
        fakebin,
        "--name",
        "KEY-12",
        "--label",
        "agent=ag-abcdefgh",
        "--label",
        "pr=1",
        "--env",
        "FOO=bar",
        "--actor",
        "auto:spawn",
        "--decision-id",
        "d-1",
    )
    assert pane["name"] == "KEY-12" and pane["labels"] == ["agent=ag-abcdefgh", "pr=1"]
    assert pane["decision_ids"] == ["d-1"] and pane["state"] == "running"
    argv = fakebin.invocations("termctl")[-1].argv
    assert argv[-3:] == ["--", "sleep", "30"] and argv[-4] == "--json"


def test_argv_after_the_first_separator_is_taken_verbatim_flags_included(fakebin: Fakebin) -> None:
    # "--json" after "--" is part of argv, not a flag: the dry run prints nothing in human mode
    done = fakebin.run("termctl", "open", "--dry-run", "--", "sh", "-c", "echo --json -- x")
    assert done.returncode == 0 and done.stdout == ""
    assert fakebin.invocations("termctl")[-1].argv == [
        "open",
        "--dry-run",
        "--",
        "sh",
        "-c",
        "echo --json -- x",
    ]


def test_open_dry_run_validates_and_changes_nothing(fakebin: Fakebin) -> None:
    code, plan = run_json(
        fakebin,
        "termctl",
        "--dry-run".replace("--dry-run", "open"),
        "--dry-run",
        "--env",
        "A=1",
        "--",
        "sleep",
        "1",
        schema="cli/open-dry-run.json",
    )
    assert code == 0 and plan["dry_run"] is True and plan["env_names"] == ["A"]
    _, listed = run_json(fakebin, "termctl", "list", schema="cli/list.json")
    assert listed["panes"] == []


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["open"], id="no-separator"),
        pytest.param(["open", "--"], id="empty-argv"),
        pytest.param(["open", "--name", "", "--", "sleep", "1"], id="empty-name"),
        pytest.param(["open", "--name", "x#[y", "--", "sleep", "1"], id="style-injection-name"),
        pytest.param(["open", "--name", "a\tb", "--", "sleep", "1"], id="control-char-name"),
        pytest.param(["open", "--name", "x" * 65, "--", "sleep", "1"], id="name-too-long"),
        pytest.param(["open", "--label", "Bad", "--", "sleep", "1"], id="bad-label"),
        pytest.param(
            ["open", "--env", "A=1", "--env", "A=2", "--", "sleep", "1"], id="duplicate-env"
        ),
        pytest.param(["open", "--env", "1A=1", "--", "sleep", "1"], id="bad-env-name"),
        pytest.param(["open", "--session", "a b", "--", "sleep", "1"], id="bad-session"),
        pytest.param(["open", "--target", "tab", "--", "sleep", "1"], id="bad-target"),
        pytest.param(["open", "--keep-on-exit", "sometimes", "--", "sleep", "1"], id="bad-keep"),
        pytest.param(["open", "--actor", "root", "--", "sleep", "1"], id="bad-actor"),
        pytest.param(
            ["open", "--hook", "pre-close=true", "--", "sleep", "1"], id="hook-of-other-command"
        ),
        pytest.param(["open", "--cwd", "/nonexistent-dir", "--", "sleep", "1"], id="missing-cwd"),
        pytest.param(["open", "--bogus", "--", "sleep", "1"], id="unknown-flag"),
        pytest.param(["--", "list"], id="separator-before-subcommand"),
        pytest.param(["show"], id="show-without-id"),
        pytest.param(["show", "tc-ab"], id="id-too-short"),
        pytest.param(["label", "add", "tc-abcdefgh"], id="label-without-pairs"),
        pytest.param(["frobnicate"], id="unknown-command"),
        pytest.param(["close", "tc-abcdefgh", "--grace", "5s"], id="grace-without-force"),
    ],
)
def test_usage_errors_exit_2_with_the_error_envelope(fakebin: Fakebin, argv: list[str]) -> None:
    # --json must come before any "--"
    at = argv.index("--") if "--" in argv else len(argv)
    done = fakebin.run("termctl", *argv[:at], "--json", *argv[at:])
    assert done.returncode == 2, done.stdout
    body = json.loads(done.stdout)
    sibling_schemas.validate(body, "termctl", "cli/error.json")
    assert body["error"]["code"] == "usage"


def test_a_missing_command_is_command_not_found_and_open_never_exits_10(fakebin: Fakebin) -> None:
    code, error = error_of(fakebin, "termctl", "open", "--", "definitely-not-a-program")
    assert (code, error["code"]) == (12, "command_not_found")
    code, error = error_of(fakebin, "termctl", "open", "--", "./nope")
    assert (code, error["code"]) == (12, "command_not_found")


def test_env_path_decides_where_argv0_is_looked_up(fakebin: Fakebin, tmp_path: Path) -> None:
    tool = tmp_path / "bin" / "mytool"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    code, _ = error_of(fakebin, "termctl", "open", "--", "mytool")
    assert code == 12
    run_json(
        fakebin,
        "termctl",
        "open",
        "--env",
        f"PATH={tool.parent}",
        "--",
        "mytool",
        schema="cli/open.json",
    )


def test_lifecycle_show_list_mark_rename_label_focus_close(fakebin: Fakebin) -> None:
    pane = open_pane(fakebin, "--name", "one", "--label", "agent=ag-1")
    other = open_pane(fakebin, "--name", "two", "--session", "other")
    pid = str(pane["id"])
    _, shown = run_json(fakebin, "termctl", "show", pid, schema="cli/show.json")
    assert shown["name"] == "one"
    _, listed = run_json(fakebin, "termctl", "list", schema="cli/list.json")
    assert [p["id"] for p in listed["panes"]] == [pane["id"], other["id"]]
    _, only = run_json(fakebin, "termctl", "list", "--label", "agent=ag-1", schema="cli/list.json")
    assert [p["id"] for p in only["panes"]] == [pane["id"]]
    _, bysession = run_json(
        fakebin, "termctl", "list", "--session", "other", schema="cli/list.json"
    )
    assert [p["id"] for p in bysession["panes"]] == [other["id"]]
    _, marked = run_json(fakebin, "termctl", "mark", pid, "attention", schema="cli/mark.json")
    assert marked["changed"] and marked["marker"] == "attention"
    _, again = run_json(fakebin, "termctl", "mark", pid, "attention", schema="cli/mark.json")
    assert again["changed"] is False and again["display_updated"] is False
    _, cleared = run_json(fakebin, "termctl", "unmark", pid, schema="cli/unmark.json")
    assert cleared["marker"] is None
    _, renamed = run_json(fakebin, "termctl", "rename", pid, "new name", schema="cli/rename.json")
    assert renamed["name"] == "new name"
    _, added = run_json(
        fakebin, "termctl", "label", "add", pid, "a=b", "c=d", schema="cli/label-add.json"
    )
    assert added["labels"] == ["a=b", "agent=ag-1", "c=d"] and added["display_updated"] is False
    _, removed = run_json(
        fakebin, "termctl", "label", "remove", pid, "a=b", schema="cli/label-remove.json"
    )
    assert removed["labels"] == ["agent=ag-1", "c=d"]
    _, dry = run_json(
        fakebin,
        "termctl",
        "label",
        "remove",
        pid,
        "c=d",
        "--dry-run",
        schema="cli/label-remove.json",
    )
    assert dry["dry_run"] and dry["labels"] == ["agent=ag-1"]
    _, kept = run_json(fakebin, "termctl", "show", pid, schema="cli/show.json")
    assert kept["labels"] == ["agent=ag-1", "c=d"]  # a dry run changed nothing
    _, focus = run_json(fakebin, "termctl", "focus", pid, schema="cli/focus.json")
    assert focus["action"] == "switched"


def test_close_rules_running_needs_force_and_force_is_human_only(fakebin: Fakebin) -> None:
    pid = str(open_pane(fakebin)["id"])
    code, error = error_of(fakebin, "termctl", "close", pid)
    assert (code, error["code"], error["reason"]) == (20, "refused", "running")
    assert error_of(fakebin, "termctl", "close", pid, "--force")[0] == 2  # no TTY, no human actor
    assert error_of(fakebin, "termctl", "close", pid, "--force", "--actor", "auto:x")[0] == 2
    assert error_of(fakebin, "termctl", "close", pid[:6], "--force", "--actor", "human:me")[0] == 2
    _, plan = run_json(
        fakebin,
        "termctl",
        "close",
        pid,
        "--force",
        "--actor",
        "human:me",
        "--dry-run",
        schema="cli/close-dry-run.json",
    )
    assert plan["forced"] and plan["signals_possible"] == ["SIGTERM", "SIGKILL"]
    _, closed = run_json(
        fakebin,
        "termctl",
        "close",
        pid,
        "--force",
        "--actor",
        "human:me",
        "--decision-id",
        "d9",
        schema="cli/close.json",
    )
    assert closed["pane_closed"] and closed["was_running"] and closed["decision_ids"] == ["d9"]
    code, error = error_of(fakebin, "termctl", "close", pid)  # already closed
    assert (code, error["code"]) == (10, "not_found")


def test_exists_prints_its_normal_output_on_exit_10(fakebin: Fakebin) -> None:
    pid = str(open_pane(fakebin)["id"])
    code, found = run_json(fakebin, "termctl", "exists", pid, schema="cli/exists.json")
    assert (code, found["exists"], found["id"]) == (0, True, pid)
    code, missing = run_json(fakebin, "termctl", "exists", "tc-aaaaaaaa", schema="cli/exists.json")
    assert (code, missing["exists"], missing["id"]) == (10, False, None)


def test_ids_prefixes_ambiguity_and_no_server(fakebin: Fakebin) -> None:
    code, error = error_of(fakebin, "termctl", "show", "tc-abcdefgh")
    assert (code, error["code"]) == (10, "no_server")
    a, b = open_pane(fakebin), open_pane(fakebin)
    prefix = str(a["id"])[3:7]
    assert str(b["id"])[3:7] == prefix  # sequential ids share a long prefix
    code, error = error_of(fakebin, "termctl", "show", prefix)
    assert (code, error["code"]) == (21, "ambiguous")
    assert [c["id"] for c in error["candidates"]] == sorted([str(a["id"]), str(b["id"])])
    code, error = error_of(fakebin, "termctl", "show", "tc-mmmmmmmm")
    assert (code, error["code"]) == (10, "not_found")


def test_servers_are_separate_per_socket(fakebin: Fakebin, tmp_path: Path) -> None:
    sock = tmp_path / "s.sock"
    pane = open_pane(fakebin, "--socket", str(sock))
    assert pane["socket_path"] == str(sock)
    _, default = run_json(fakebin, "termctl", "list", schema="cli/list.json")
    assert default["panes"] == []
    _, named = run_json(fakebin, "termctl", "list", "--socket", str(sock), schema="cli/list.json")
    assert len(named["panes"]) == 1
    _, byname = run_json(fakebin, "termctl", "list", "--socket", "work", schema="cli/list.json")
    assert byname["socket_path"].endswith("/work") and "/tmux-" in byname["socket_path"]


def test_global_flags_before_or_after_the_subcommand_and_version(fakebin: Fakebin) -> None:
    code, listed = run_json(fakebin, "termctl", "--socket", "w", "list", schema="cli/list.json")
    assert code == 0
    done = fakebin.run("termctl", "--json", "list")
    assert done.returncode == 0 and json.loads(done.stdout)["panes"] == []
    version = fakebin.run("termctl", "list", "--bogus", "--version")
    assert (version.returncode, version.stdout) == (0, "termctl 0.1.0\n")  # --version wins
    assert fakebin.run("termctl", "--json=x", "list").returncode == 2


def test_hanging_close_can_be_scripted_for_signal_tests(fakebin: Fakebin) -> None:
    pid = str(open_pane(fakebin)["id"])
    fakebin.script("termctl", "close", error={"code": "locked", "message": "busy"}, exit=12)
    code, error = error_of(fakebin, "termctl", "close", pid, "--force", "--actor", "human:me")
    assert (code, error["code"]) == (12, "locked")
    _, still = run_json(fakebin, "termctl", "exists", pid, schema="cli/exists.json")
    assert still["exists"] is True  # a scripted error replaces the output; nothing was closed


def test_placements_is_not_implemented(fakebin: Fakebin) -> None:
    done = fakebin.run("termctl", "placements", "--json")
    assert done.returncode == 1 and json.loads(done.stdout)["error"]["code"] == "fake_unsupported"


# --- global flags with a separate value before a nested command (R2) --------------------------


@pytest.mark.parametrize("before", [True, False], ids=["flag-before-label", "flag-after-label"])
def test_label_add_accepts_a_global_value_flag_in_either_position(
    fakebin: Fakebin, tmp_path: Path, before: bool
) -> None:
    sock = str(tmp_path / "s")
    pane = open_pane(fakebin, "--socket", sock)
    flag = ["--socket", sock]
    argv = (
        [*flag, "label", "add", str(pane["id"]), "k=v"]
        if before
        else ["label", "add", *flag, str(pane["id"]), "k=v"]
    )
    code, body = run_json(fakebin, "termctl", *argv, schema="cli/label-add.json")
    assert code == 0 and "k=v" in body["labels"]


def test_label_remove_with_a_config_flag_value_named_like_an_action(fakebin: Fakebin) -> None:
    pane = open_pane(fakebin, "--label", "k=v")
    code, body = run_json(
        fakebin, "termctl", "--config", "add", "label", "remove", str(pane["id"]), "k=v"
    )
    assert code == 0 and body["labels"] == []


# --- limits and checks the real termctl makes (R11) -------------------------------------------


def test_label_add_beyond_32_labels_exits_2_after_resolving_the_id(fakebin: Fakebin) -> None:
    pane = open_pane(fakebin)
    pairs = [f"k{i}=v" for i in range(33)]
    fits = fakebin.run("termctl", "label", "add", str(pane["id"]), *pairs[:32], "--json")
    assert fits.returncode == 0  # 32 fit
    done = fakebin.run("termctl", "label", "add", str(pane["id"]), *pairs, "--json")
    assert done.returncode == 2
    unknown = fakebin.run("termctl", "label", "add", "tc-zzzzzzzz", *pairs, "--json")
    assert unknown.returncode == 10


def test_open_with_an_empty_cwd_or_an_empty_command_is_refused(fakebin: Fakebin) -> None:
    assert fakebin.run("termctl", "open", "--cwd", "", "--json", "--", "sleep", "1").returncode == 2
    done = fakebin.run("termctl", "open", "--json", "--", "")
    assert done.returncode == 12 and json.loads(done.stdout)["error"]["code"] == "command_not_found"
    assert fakebin.run("termctl", "open", "--json", "--", "A=1", "sleep").returncode == 2


@pytest.mark.parametrize("command", ["open", "close"])
def test_override_escalation_needs_a_human_actor_or_a_terminal(
    fakebin: Fakebin, command: str
) -> None:
    pane = str(open_pane(fakebin)["id"])
    base = ["open", "--dry-run", "--", "sleep", "1"] if command == "open" else ["close", pane]

    def call(*extra: str) -> int:
        head, tail = (base[:1], base[1:]) if command == "close" else (base[:2], base[2:])
        return fakebin.run(
            "termctl", *head, "--json", "--override-escalation", *extra, *tail
        ).returncode

    assert call() == 2
    assert call("--actor", "auto:cron") == 2
    assert call("--actor", "human:me") in (0, 20)  # close of a running pane is refused (20)


def test_mark_with_an_undefined_marker_exits_2_unless_it_is_built_in_or_scripted(
    fakebin: Fakebin,
) -> None:
    pane = str(open_pane(fakebin)["id"])
    assert fakebin.run("termctl", "mark", pane, "nosuch", "--json").returncode == 2
    code, body = run_json(fakebin, "termctl", "mark", pane, "attention", schema="cli/mark.json")
    assert code == 0 and body["marker"] == "attention" and body["marker_text"] == "\U0001f514 "
    fakebin.script("termctl", "mark", markers={"review": "R "})
    code, body = run_json(fakebin, "termctl", "mark", pane, "review", schema="cli/mark.json")
    assert (code, body["marker_text"]) == (0, "R ")


def test_the_default_session_is_termctl(fakebin: Fakebin) -> None:
    assert open_pane(fakebin)["session"] == "termctl"
    _, plan = run_json(
        fakebin, "termctl", "open", "--dry-run", "--", "sleep", "1", schema="cli/open-dry-run.json"
    )
    assert plan["session"] == "termctl"


def test_spellings_of_one_socket_path_share_state(fakebin: Fakebin, tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    pane = open_pane(fakebin, "--socket", str(real / "sock"))
    for spelling in (
        str(tmp_path / "link" / "sock"),
        str(tmp_path) + "//real//sock",
        str(tmp_path / "real" / "x" / ".." / "sock"),
    ):
        # `x` does not exist: a lexical `..` still names the same socket
        code, body = run_json(fakebin, "termctl", "--socket", spelling, "show", str(pane["id"]))
        assert code == 0 and body["id"] == pane["id"], spelling
