"""Agent ids: grammar, allocation (never reused) and prefix resolution."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from agent_spawn import ids
from agent_spawn.ids import IdEntry
from agent_spawn.output import CliError
from tests import schema_registry


class Index:
    """A tiny in-memory stand-in for the state store's lookup.

    ``reserved`` are ids allocated once whose record is gone (a deleted tombstone or fork record).
    """

    def __init__(self, *entries: IdEntry, reserved: Sequence[str] = ()) -> None:
        self.entries = list(entries)
        self.reserved = set(reserved)
        self.queries: list[str] = []

    def entries_with_prefix(self, prefix: str) -> Sequence[IdEntry]:
        self.queries.append(prefix)
        return [entry for entry in self.entries if entry.agent_id.startswith(prefix)]

    def is_reserved(self, agent_id: str) -> bool:
        return agent_id in self.reserved or any(e.agent_id == agent_id for e in self.entries)


def live(agent_id: str) -> IdEntry:
    return IdEntry(agent_id, torn_down=False)


def gone(agent_id: str) -> IdEntry:
    return IdEntry(agent_id, torn_down=True)


# --- grammar and allocation --------------------------------------------------------------------


@pytest.mark.parametrize("value", ["ag-abcdefgh", "ag-22222222", "ag-77777777", "ag-a2b7c3d4"])
def test_full_ids_match(value: str) -> None:
    assert ids.is_agent_id(value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "ag-",
        "ag-abcdefg",
        "ag-abcdefghi",
        "ag-ABCDEFGH",
        "ag-abcdefg1",  # 1 is not base32
        "ag-abcdefg8",
        "ag-abcdefg0",
        "ag-abcdefg\n",
        "ag-abcdefgh\n",
        "\nag-abcdefgh",
        "AG-abcdefgh",
        "abcdefgh",
        "ag_abcdefgh",
        "ag-abcdefgа",
        "ag-abcdef\x00h",
        " ag-abcdefgh",
    ],
)
def test_non_ids_do_not_match(value: str) -> None:
    assert not ids.is_agent_id(value)


def test_new_agent_id_has_the_grammar() -> None:
    for _ in range(200):
        assert ids.is_agent_id(ids.new_agent_id(Index()))


def test_new_agent_id_uses_the_whole_alphabet_over_many_draws() -> None:
    seen = {ch for _ in range(2000) for ch in ids.new_agent_id(Index())[3:]}
    assert seen == set("abcdefghijklmnopqrstuvwxyz234567")


def test_new_agent_id_never_reuses_a_live_or_tombstoned_id() -> None:
    index = Index(live("ag-aaaaaaaa"), gone("ag-77777777"))
    draws = iter([b"\x00" * 5, b"\xff" * 5, b"\x01" * 5])
    result = ids.new_agent_id(index, random_bytes=lambda n: next(draws))
    assert ids.is_agent_id(result)
    assert result not in {"ag-aaaaaaaa", "ag-77777777"}


def test_new_agent_id_never_reuses_a_reserved_id_without_a_record() -> None:
    index = Index(reserved=["ag-aaaaaaaa"])
    draws = iter([b"\x00" * 5, b"\x00" * 5, b"\xff" * 5])
    assert ids.new_agent_id(index, random_bytes=lambda n: next(draws)) == "ag-77777777"
    assert index.queries == []  # allocation asks is_reserved, not the record lookup


def test_new_agent_id_gives_up_after_bounded_collisions() -> None:
    index = Index(live("ag-aaaaaaaa"))
    with pytest.raises(RuntimeError):
        ids.new_agent_id(index, random_bytes=lambda n: b"\x00" * n)


# --- prefix resolution -------------------------------------------------------------------------

A = "ag-abcdefgh"
B = "ag-abcdwxyz"
C = "ag-q2222222"


def resolve(text: str, *entries: IdEntry, reserved: Sequence[str] = ()) -> ids.ResolvedId:
    return ids.resolve_agent_id(text, Index(*entries, reserved=reserved))


def error_of(text: str, *entries: IdEntry, reserved: Sequence[str] = ()) -> CliError:
    with pytest.raises(CliError) as info:
        resolve(text, *entries, reserved=reserved)
    return info.value


@pytest.mark.parametrize("text", [A, "abcdefgh"])
def test_full_id_resolves_with_or_without_the_prefix(text: str) -> None:
    assert resolve(text, live(A), live(C)) == ids.ResolvedId(A, torn_down=False)


@pytest.mark.parametrize("text", ["ag-abce", "abce"])
def test_unique_prefix_of_four_or_more_resolves(text: str) -> None:
    assert resolve(text, live("ag-abcefghi"), live(B)).agent_id == "ag-abcefghi"


def test_prefix_lengths_four_to_seven_resolve() -> None:
    for length in range(4, 8):
        assert resolve(A[3 : 3 + length], live(A), live(C)).agent_id == A


@pytest.mark.parametrize("text", ["", "ag-", "a", "ag-a", "ag-ab", "ag-abc", "abc", "ag-ag-abcd"])
def test_fewer_than_four_characters_exit_2(text: str) -> None:
    err = error_of(text, live(A))
    assert (err.code, err.exit_code) == ("usage", 2)


@pytest.mark.parametrize(
    "text",
    ["ag-abcdefghi", "ag-ABCD", "ag-abc1", "ag-ab cd", "ag-abcd\n", "ag-abcd\x00", "ag-аbcd"],
)
def test_malformed_ids_exit_2(text: str) -> None:
    assert error_of(text, live(A)).exit_code == 2


def test_no_match_exits_10() -> None:
    err = error_of("ag-zzzz", live(A))
    assert (err.code, err.exit_code) == ("not_found", 10)


def test_unknown_full_id_exits_10() -> None:
    assert error_of("ag-zzzzzzzz", live(A)).exit_code == 10


def test_several_matches_exit_21_with_sorted_candidates() -> None:
    err = error_of("ag-abcd", live(B), live(A))
    assert (err.code, err.exit_code) == ("ambiguous", 21)
    assert err.candidates == [A, B]


def test_ambiguous_prefix_excludes_torn_down_agents_from_candidates_and_the_count() -> None:
    err = error_of("ag-abcd", live(A), live(B), gone("ag-abcdaaaa"))
    assert err.candidates == [A, B]


def test_torn_down_agents_never_match_a_prefix() -> None:
    err = error_of("ag-abcd", gone(A))
    assert err.exit_code == 10


def test_a_prefix_unique_among_live_agents_ignores_tombstones() -> None:
    assert resolve("ag-abcd", live(A), gone("ag-abcdaaaa")).agent_id == A


def test_full_id_of_a_torn_down_agent_resolves_and_says_so() -> None:
    assert resolve(A, gone(A)) == ids.ResolvedId(A, torn_down=True)
    assert resolve("abcdefgh", gone(A), live(B)) == ids.ResolvedId(A, torn_down=True)


def test_full_id_reserved_without_a_record_exits_10_without_torn_down() -> None:
    err = error_of(A, live(B), reserved=[A])
    assert (err.code, err.exit_code) == ("not_found", 10)
    assert err.details is None


def test_a_prefix_never_matches_a_reserved_id_without_a_record() -> None:
    assert error_of("ag-abcd", reserved=[A]).exit_code == 10
    assert resolve("ag-abcd", live(B), reserved=[A]).agent_id == B


@pytest.mark.parametrize(
    ("text", "code"), [("ag-ab", "usage"), ("ag-zzzz", "not_found"), ("ag-abcd", "ambiguous")]
)
def test_resolution_errors_fit_the_error_envelope_schema(text: str, code: str) -> None:
    err = error_of(text, live(A), live(B))
    assert err.code == code
    schema_registry.validate(err.envelope(), "agent-spawn/cli/error.json")


def test_full_id_is_never_ambiguous() -> None:
    # No id is a prefix of another (all are 8 chars), but a full id must still resolve exactly.
    assert resolve(A, live(A), live(B)).agent_id == A


def test_not_found_message_does_not_echo_nothing_untrusted() -> None:
    err = error_of("ag-zzzz", live(A))
    assert "zzzz" in err.message  # the id grammar is safe ASCII once it has been checked


def test_malformed_message_escapes_control_characters_and_is_bounded() -> None:
    err = error_of("ag-\x1b[31m" + "z" * 5000, live(A))
    assert "\x1b" not in err.message
    assert len(err.message) < 300


def test_resolution_queries_the_index_with_the_prefix_without_ag() -> None:
    index = Index(live(A))
    ids.resolve_agent_id("ag-abcd", index)
    assert index.queries == ["ag-abcd"]
    index.queries.clear()
    ids.resolve_agent_id("abcd", index)
    assert index.queries == ["ag-abcd"]


def test_resolution_trusts_the_index_only_for_ids_that_start_with_the_prefix() -> None:
    class Sloppy:
        def entries_with_prefix(self, prefix: str) -> Sequence[IdEntry]:
            return [live(A), live(C)]

        def is_reserved(self, agent_id: str) -> bool:
            return True

    assert ids.resolve_agent_id("ag-abcd", Sloppy()).agent_id == A


def test_agent_id_argument_type_validates_grammar_only() -> None:
    import argparse

    assert ids.agent_id_arg("ag-abcd") == "ag-abcd"
    assert ids.agent_id_arg("abcdefgh") == "abcdefgh"
    for bad in ("ag-ab", "", "ag-ABCD", "ag-abcdefghi"):
        with pytest.raises(argparse.ArgumentTypeError):
            ids.agent_id_arg(bad)
