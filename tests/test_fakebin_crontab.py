"""The fake crontab: ``-l``, ``-`` and ``-r`` on a table that never leaves the test's directory."""

from __future__ import annotations

import pytest

from tests.fakebin.fakelib.crontab import check_table
from tests.fakebin.harness import Fakebin

JOB = "*/5 * * * * /bin/true\n"


def test_without_a_table_list_and_remove_say_there_is_none(fakebin: Fakebin) -> None:
    for flag in ("-l", "-r"):
        done = fakebin.run("crontab", flag)
        assert (done.returncode, done.stdout) == (1, "")
        assert done.stderr == "no crontab for fakeuser\n"


def test_install_replaces_the_whole_table_and_list_prints_it_back(fakebin: Fakebin) -> None:
    assert fakebin.run("crontab", "-", input="# a\n" + JOB).returncode == 0
    assert fakebin.run("crontab", "-l").stdout == "# a\n" + JOB
    assert fakebin.run("crontab", "-", input="MAILTO=x\n").returncode == 0
    assert fakebin.run("crontab", "-l").stdout == "MAILTO=x\n"
    assert fakebin.run("crontab", "-r").returncode == 0
    assert fakebin.run("crontab", "-l").returncode == 1


def test_the_table_is_stored_next_to_the_log_only(fakebin: Fakebin) -> None:
    fakebin.run("crontab", "-", input=JOB)
    assert (fakebin.root / "invocations.jsonl.crontab").read_text() == JOB


def test_a_table_the_real_crontab_would_refuse_changes_nothing(fakebin: Fakebin) -> None:
    fakebin.run("crontab", "-", input=JOB)
    for bad in (JOB.rstrip("\n"), "not a job\n", "* * * * *\n"):
        done = fakebin.run("crontab", "-", input=bad)
        assert done.returncode == 1 and "can't install" in done.stderr
    assert fakebin.run("crontab", "-l").stdout == JOB


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("", True),
        ("\n# c\n  \n", True),
        ("PATH=/x:/y\n", True),
        ("@reboot /bin/true\n", True),
        ("@reboot\n", False),
        ("* * * * * cmd\n", True),
        ("* * * * *\n", False),
        ("* * * * * cmd", False),
    ],
)
def test_check_table(text: str, ok: bool) -> None:
    assert (check_table(text) is None) is ok


def test_a_scripted_call_leaves_the_table_alone_and_logs_stdin(fakebin: Fakebin) -> None:
    fakebin.script("crontab", "install", exit=1, stderr="crontab: no space\n")
    done = fakebin.run("crontab", "-", input=JOB)
    assert (done.returncode, done.stderr) == (1, "crontab: no space\n")
    assert fakebin.run("crontab", "-l").returncode == 1
    assert fakebin.invocations("crontab")[0].stdin == JOB
    fakebin.script("crontab", "list", stdout=JOB)
    assert fakebin.run("crontab", "-l").stdout == JOB


def test_other_options_are_refused_never_silently_accepted(fakebin: Fakebin) -> None:
    assert fakebin.run("crontab", "-e").returncode == 1
    assert fakebin.run("crontab", "-u", "bob", "-l").returncode == 1
    assert fakebin.run("crontab", "-x").returncode == 1
    assert fakebin.run("crontab").returncode == 1
