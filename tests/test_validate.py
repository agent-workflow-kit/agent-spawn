"""Shared input grammars: durations, names, actors, decision ids, merged-head shas, times."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agent_spawn import validate
from agent_spawn.validate import GrammarError

NOW = datetime(2026, 11, 3, 12, 0, 0, tzinfo=UTC)


# --- durations ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("1", 1.0),
        ("90", 90.0),
        ("0.5", 0.5),
        ("007", 7.0),
        ("1500ms", 1.5),
        ("30s", 30.0),
        ("5m", 300.0),
        ("2h", 7200.0),
        ("1d", 86400.0),
        ("1.5h", 5400.0),
        ("0.25s", 0.25),
        ("1.0d", 86400.0),
    ],
)
def test_duration_accepts(text: str, seconds: float) -> None:
    assert validate.parse_duration(text) == seconds


@pytest.mark.parametrize(
    "text",
    [
        "",
        "0",
        "0s",
        "0.0",
        "0.000ms",
        "00",
        "-1",
        "-1s",
        "+1",
        "+1s",
        " 1",
        "1 ",
        "1 s",
        "1\n",
        "\n1",
        "1s\n",
        "1\x00",
        ".5",
        "5.",
        "1.2.3",
        "1e3",
        "1E3",
        "inf",
        "nan",
        "Infinity",
        "NaN",
        "5x",
        "5S",
        "5M",
        "5min",
        "m",
        "s",
        "ms",
        "1_000",
        "0x10",
        "１",  # fullwidth digit one
        "١",  # arabic-indic digit one
        "1 s",  # thin space
        "1²",  # superscript two
        "5ѕ",  # cyrillic dze, a lookalike of s
        "9" * 40,
        "9" * 40 + "d",
        "1" + "0" * 30 + "s",
        "999999999999d",
        "1." + "0" * 40 + "1",
        "x" * 10_000,
    ],
)
def test_duration_rejects(text: str) -> None:
    with pytest.raises(GrammarError):
        validate.parse_duration(text)


def test_duration_error_does_not_echo_control_characters_or_overlong_text() -> None:
    hostile = "\x1b[31m" + "A" * 5000 + "‮"
    with pytest.raises(GrammarError) as info:
        validate.parse_duration(hostile)
    message = str(info.value)
    assert "\x1b" not in message
    assert "‮" not in message
    assert len(message) < 300


def test_duration_error_survives_lone_surrogates() -> None:
    with pytest.raises(GrammarError) as info:
        validate.parse_duration("\udcff")
    assert str(info.value).isascii()


def test_duration_zero_message_says_zero() -> None:
    with pytest.raises(GrammarError, match="greater than zero"):
        validate.parse_duration("0")


def test_duration_upper_bound_is_ten_years() -> None:
    ten_years = 10 * 365 * 86400
    assert validate.parse_duration(f"{ten_years}") == ten_years
    with pytest.raises(GrammarError):
        validate.parse_duration(f"{ten_years + 1}")


def test_duration_argparse_type_returns_the_raw_string() -> None:
    assert validate.duration_arg("5m") == "5m"
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        validate.duration_arg("0")


# --- names -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "a",
        "my agent",
        "x" * 64,
        "é" * 64,  # 64 code points, 128 bytes
        "\U0001f600" * 64,  # astral code points count once
        "a#b",
        "a[b",
        "[#",
        "tab sep",  # line separator is not a control character here
        " nbsp",
        "​zwsp",
        "café 日本語",
    ],
)
def test_name_accepts(name: str) -> None:
    assert validate.validate_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "x" * 65,
        "\U0001f600" * 65,
        "a\x00b",
        "a\nb",
        "a\rb",
        "a\tb",
        "a\x1bb",
        "a\x1fb",
        "a\x7fb",
        "a\x80b",
        "a\x85b",  # NEL, a C1 control
        "a\x9fb",
        "a#[b",
        "#[",
        "x#[",
        "\udcff",  # lone surrogate from undecodable argv
        "a\ud800b",
    ],
)
def test_name_rejects(name: str) -> None:
    with pytest.raises(GrammarError):
        validate.validate_name(name)


def test_name_rule_boundary_is_c1_end() -> None:
    assert validate.validate_name("a ") == "a "  # U+00A0 is just above C1
    with pytest.raises(GrammarError):
        validate.validate_name("a\u009f")


def test_name_error_does_not_echo_control_characters() -> None:
    with pytest.raises(GrammarError) as info:
        validate.validate_name("\x1b[2J" + "z" * 500)
    assert "\x1b" not in str(info.value)
    assert len(str(info.value)) < 300


def test_default_name_prefers_name_then_worktree_name_then_kind_and_id() -> None:
    agent_id = "ag-abcdefgh"
    assert validate.default_name(agent_id, name="n", worktree_name="w", kind="k") == "n"
    assert validate.default_name(agent_id, worktree_name="w", kind="k") == "w"
    assert validate.default_name(agent_id, kind="review") == "review abcdef"
    assert validate.default_name(agent_id) == "agent abcdef"


def test_default_name_ignores_nothing_but_none() -> None:
    # An explicit empty --name is invalid, not "unset".
    with pytest.raises(GrammarError):
        validate.default_name("ag-abcdefgh", name="", kind="k")


def test_default_name_validates_the_chosen_candidate() -> None:
    with pytest.raises(GrammarError):
        validate.default_name("ag-abcdefgh", worktree_name="bad#[name")
    with pytest.raises(GrammarError):
        validate.default_name("ag-abcdefgh", kind="x" * 80)
    with pytest.raises(GrammarError):
        validate.default_name("ag-abcdefgh", kind="a\nb")


def test_default_name_rejects_a_malformed_agent_id() -> None:
    with pytest.raises(GrammarError):
        validate.default_name("abcdefgh")


# --- actors and decision ids -------------------------------------------------------------------


@pytest.mark.parametrize(
    "actor", ["human:a", "auto:unknown", "human:a.b_c@d-e", "auto:" + "x" * 64, "human:" + "0" * 64]
)
def test_actor_accepts(actor: str) -> None:
    assert validate.validate_actor(actor) == actor


@pytest.mark.parametrize(
    "actor",
    [
        "",
        "human:",
        "human",
        "bob",
        "robot:bob",
        "Human:bob",
        "human:" + "x" * 65,
        "human:a b",
        "human:a\n",
        "human:a\x00",
        "human:а",  # cyrillic a
        "human:a:b",
        "human:a/b",
        " human:a",
        "human: a",
        "human:a\n" + "auto:b",
    ],
)
def test_actor_rejects(actor: str) -> None:
    with pytest.raises(GrammarError):
        validate.validate_actor(actor)


@pytest.mark.parametrize("value", ["a", "a:b/c@d.e_f-g", "x" * 128, "A1"])
def test_decision_id_accepts(value: str) -> None:
    assert validate.validate_decision_id(value) == value


@pytest.mark.parametrize(
    "value", ["", "x" * 129, "has space", "a\n", "a\x00", "а", "a#b", "a\tb", "a,b"]
)
def test_decision_id_rejects(value: str) -> None:
    with pytest.raises(GrammarError):
        validate.validate_decision_id(value)


def test_decision_id_list_is_capped_at_32() -> None:
    validate.validate_decision_ids(["d"] * 32)
    with pytest.raises(GrammarError):
        validate.validate_decision_ids(["d"] * 33)
    with pytest.raises(GrammarError):
        validate.validate_decision_ids(["d", "bad id"])


# --- merged-head sha ---------------------------------------------------------------------------


def test_merged_head_accepts_40_lowercase_hex() -> None:
    assert validate.validate_merged_head("0123456789abcdef" * 2 + "01234567") == (
        "0123456789abcdef" * 2 + "01234567"
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "a" * 39,
        "a" * 41,
        "a" * 64,  # a sha-256 id is not accepted
        "A" * 40,
        "g" * 40,
        "a" * 40 + "\n",
        "\n" + "a" * 40,
        " " + "a" * 40,
        "a" * 39 + "а",
        "HEAD",
        "main",
        "a" * 20 + "\x00" + "a" * 19,
    ],
)
def test_merged_head_rejects(value: str) -> None:
    with pytest.raises(GrammarError):
        validate.validate_merged_head(value)


# --- --since -----------------------------------------------------------------------------------


def test_since_duration_is_that_long_ago() -> None:
    assert validate.parse_since("2h", NOW) == NOW - timedelta(hours=2)
    assert validate.parse_since("90", NOW) == NOW - timedelta(seconds=90)


def test_since_bare_digits_are_seconds_not_a_date() -> None:
    assert validate.parse_since("20240101", NOW) == NOW - timedelta(seconds=20240101)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-11-01", datetime(2026, 11, 1, tzinfo=UTC)),
        ("2026-11-01T10:30", datetime(2026, 11, 1, 10, 30, tzinfo=UTC)),
        ("2026-11-01T10:30:15", datetime(2026, 11, 1, 10, 30, 15, tzinfo=UTC)),
        ("2026-11-01 10:30:15", datetime(2026, 11, 1, 10, 30, 15, tzinfo=UTC)),
        ("2026-11-01T10:30:15Z", datetime(2026, 11, 1, 10, 30, 15, tzinfo=UTC)),
        ("2026-11-01T10:30:15.250Z", datetime(2026, 11, 1, 10, 30, 15, 250000, tzinfo=UTC)),
        ("2026-11-01T10:30:15+02:00", datetime(2026, 11, 1, 8, 30, 15, tzinfo=UTC)),
        ("2026-11-01T10:30:15-05:30", datetime(2026, 11, 1, 16, 0, 15, tzinfo=UTC)),
    ],
)
def test_since_iso_time_is_utc_without_an_offset(text: str, expected: datetime) -> None:
    result = validate.parse_since(text, NOW)
    assert result == expected
    assert result.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "yesterday",
        "2026-13-01",
        "2026-02-30",
        "2026-11-01T25:00",
        "2026-11-01T10:30:15+25:00",
        "2026-11-01T10:30:15\n",
        "2026-11-01T",
        "20261101T1030",
        "2026-11-01T10:30:15 +02:00",
        "２026-11-01",
        "0",
        "0m",
        "-1h",
        "0000-01-01",
        "9999-12-31T23:59:59-23:59",
        "2026-11-01" + "x" * 5000,
    ],
)
def test_since_rejects(text: str) -> None:
    with pytest.raises(GrammarError):
        validate.parse_since(text, NOW)


def test_since_duration_beyond_the_epoch_range_is_rejected_not_an_overflow() -> None:
    far = datetime(1, 1, 2, tzinfo=UTC)
    with pytest.raises(GrammarError):
        validate.parse_since("1d", far)
