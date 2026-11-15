"""Label grammar, multiset semantics, the 64-pair cap and the reserved keys."""

from __future__ import annotations

import pytest

from agent_spawn import labels
from agent_spawn.labels import Label, LabelSet
from agent_spawn.validate import GrammarError


def pairs(*texts: str) -> list[Label]:
    return [labels.parse_label(text) for text in texts]


# --- grammar -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "key", "value"),
    [
        ("pr=1", "pr", "1"),
        ("a=", "a", ""),
        ("0=0", "0", "0"),
        ("k=a=b", "k", "a=b"),  # split at the first =
        ("k==", "k", "="),
        ("a.b_c-d=v", "a.b_c-d", "v"),
        ("k=" + "x" * 256, "k", "x" * 256),
        ("k" * 64 + "=v", "k" * 64, "v"),
        ("k=A.b_C:d/e#f@g+h=i,j-k", "k", "A.b_C:d/e#f@g+h=i,j-k"),
    ],
)
def test_label_accepts(text: str, key: str, value: str) -> None:
    assert labels.parse_label(text) == Label(key, value)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "=",
        "=v",
        "novalue",
        "K=v",  # uppercase key
        "_a=v",
        ".a=v",
        "-a=v",
        "a b=v",
        "a=b c",
        "a=b\n",
        "a=b\x00",
        "a\n=b",
        "a=а",  # cyrillic a
        "а=b",
        "a=é",
        "a=\U0001f600",
        "a=b;c",
        "a=b$c",
        "a=b%c",
        "a=b\\c",
        'a=b"c',
        "a=b'c",
        "a=b?c",
        "k" * 65 + "=v",
        "k=" + "x" * 257,
        "a＝b",  # fullwidth equals sign is not a separator
        "a=‮",
        "\udcff=a",
        "a=\udcff",
    ],
)
def test_label_rejects(text: str) -> None:
    with pytest.raises(GrammarError):
        labels.parse_label(text)


def test_label_error_does_not_echo_untrusted_text_raw() -> None:
    with pytest.raises(GrammarError) as info:
        labels.parse_label("a=\x1b[31m" + "z" * 4000)
    assert "\x1b" not in str(info.value)
    assert len(str(info.value)) < 400


def test_label_text_roundtrip() -> None:
    assert str(Label("a", "b=c")) == "a=b=c"


@pytest.mark.parametrize("key", ["agent", "kind"])
def test_reserved_keys_are_rejected_for_user_labels(key: str) -> None:
    with pytest.raises(GrammarError, match="reserved"):
        labels.parse_user_label(f"{key}=x")
    with pytest.raises(GrammarError, match="reserved"):
        labels.parse_user_label(f"{key}=")


def test_reserved_check_is_by_exact_key() -> None:
    assert labels.parse_user_label("agents=x") == Label("agents", "x")
    assert labels.parse_user_label("kind2=x") == Label("kind2", "x")
    assert labels.parse_user_label("x=agent") == Label("x", "agent")


def test_parse_label_allows_reserved_keys_for_internal_pairs() -> None:
    assert labels.parse_label("agent=ag-abcdefgh") == Label("agent", "ag-abcdefgh")
    assert labels.parse_label("kind=review") == Label("kind", "review")


def test_selector_allows_reserved_keys_but_not_bad_grammar() -> None:
    assert labels.parse_selector("kind=x") == Label("kind", "x")
    with pytest.raises(GrammarError):
        labels.parse_selector("Kind=x")


def test_argparse_types_return_the_raw_text() -> None:
    import argparse

    assert labels.label_arg("a=b") == "a=b"
    assert labels.selector_arg("kind=b") == "kind=b"
    with pytest.raises(argparse.ArgumentTypeError):
        labels.label_arg("agent=b")
    with pytest.raises(argparse.ArgumentTypeError):
        labels.label_arg("nope")
    with pytest.raises(argparse.ArgumentTypeError):
        labels.selector_arg("nope")


# --- multiset ----------------------------------------------------------------------------------


def test_a_key_may_carry_several_values() -> None:
    result = LabelSet().add(pairs("job=1", "job=2")).labels
    assert result.pairs == (Label("job", "1"), Label("job", "2"))


def test_adding_an_existing_pair_is_a_noop() -> None:
    start = LabelSet().add(pairs("a=1", "b=2")).labels
    change = start.add(pairs("a=1"))
    assert change.changed is False
    assert change.applied == ()
    assert change.labels == start


def test_adding_reports_only_the_pairs_actually_added_in_order() -> None:
    start = LabelSet().add(pairs("a=1")).labels
    change = start.add(pairs("b=2", "a=1", "c=3", "b=2"))
    assert change.applied == (Label("b", "2"), Label("c", "3"))
    assert change.changed is True
    assert change.labels.pairs == (Label("a", "1"), Label("b", "2"), Label("c", "3"))


def test_stored_order_is_insertion_order() -> None:
    result = LabelSet().add(pairs("z=1", "a=1", "m=1")).labels
    assert [pair.key for pair in result.pairs] == ["z", "a", "m"]


def test_removing_a_missing_pair_is_a_noop() -> None:
    start = LabelSet().add(pairs("a=1")).labels
    change = start.remove(pairs("a=2", "b=1"))
    assert change.changed is False
    assert change.applied == ()
    assert change.labels == start


def test_removing_removes_exactly_that_pair() -> None:
    start = LabelSet().add(pairs("job=1", "job=2", "a=1")).labels
    change = start.remove(pairs("job=1", "job=1", "x=1"))
    assert change.applied == (Label("job", "1"),)
    assert change.labels.pairs == (Label("job", "2"), Label("a", "1"))


def test_empty_value_is_distinct_from_missing_value() -> None:
    start = LabelSet().add(pairs("a=")).labels
    assert start.remove(pairs("a=x")).changed is False
    assert start.remove(pairs("a=")).labels.pairs == ()


def test_label_set_is_immutable() -> None:
    start = LabelSet().add(pairs("a=1")).labels
    start.add(pairs("b=1"))
    start.remove(pairs("a=1"))
    assert start.pairs == (Label("a", "1"),)


def test_to_dicts_and_text_forms() -> None:
    start = LabelSet().add(pairs("a=1", "b=")).labels
    assert start.texts() == ["a=1", "b="]
    assert Label("a", "1") in start


def test_from_texts_rebuilds_a_stored_set_and_collapses_duplicates() -> None:
    assert LabelSet.from_texts(["a=1", "a=1", "b=2"]).texts() == ["a=1", "b=2"]
    with pytest.raises(GrammarError):
        LabelSet.from_texts(["bad"])


# --- the 64-pair cap ---------------------------------------------------------------------------


def full_set(count: int = labels.MAX_LABELS) -> LabelSet:
    return LabelSet().add([Label("k", str(index)) for index in range(count)]).labels


def test_exactly_64_pairs_fit() -> None:
    assert len(full_set()) == 64


def test_the_65th_distinct_pair_is_refused() -> None:
    with pytest.raises(GrammarError, match="64"):
        full_set().add(pairs("extra=1"))


def test_a_refused_add_changes_nothing_in_the_set() -> None:
    start = full_set(63)
    with pytest.raises(GrammarError):
        start.add(pairs("x=1", "x=2"))
    assert len(start) == 63


def test_duplicates_do_not_count_toward_the_cap() -> None:
    start = full_set()
    assert start.add(pairs("k=0", "k=1")).changed is False
    assert len(LabelSet().add([Label("k", "1")] * 100).labels) == 1


def test_remove_then_add_fits_across_calls() -> None:
    start = full_set()
    shrunk = start.remove([Label("k", "0")]).labels
    assert len(shrunk.add(pairs("extra=1")).labels) == 64


def test_add_job_beyond_the_cap_is_skipped_not_an_error() -> None:
    start = full_set()
    change = start.add_job(labels.parse_label("job=abc"))
    assert change.changed is False
    assert change.skipped is True
    assert change.labels == start


def test_add_job_within_the_cap_adds_and_is_idempotent() -> None:
    start = full_set(10)
    first = start.add_job(labels.parse_label("job=abc"))
    assert first.changed is True and first.skipped is False
    again = first.labels.add_job(labels.parse_label("job=abc"))
    assert again.changed is False and again.skipped is False


def test_add_job_only_takes_job_keys() -> None:
    with pytest.raises(ValueError, match="job"):
        LabelSet().add_job(Label("a", "1"))


def test_job_label_from_environment_value_must_fit_the_value_grammar() -> None:
    assert labels.job_label("abc-123") == Label("job", "abc-123")
    assert labels.job_label("") == Label("job", "")
    assert labels.job_label("has space") is None
    assert labels.job_label("x" * 257) is None
    assert labels.job_label("a\n") is None


def test_parse_user_labels_checks_the_cap_over_distinct_pairs() -> None:
    texts = [f"k={index}" for index in range(64)]
    assert len(labels.parse_user_labels(texts + texts)) == 64
    with pytest.raises(GrammarError, match="64"):
        labels.parse_user_labels([*texts, "k=64"])


# --- labels given at launch and resume (job= beyond the cap is skipped) ------------------------


def test_run_labels_skip_a_job_pair_beyond_the_cap() -> None:
    change = LabelSet().add_run([*full_set().pairs, Label("job", "x")])
    assert len(change.labels) == 64
    assert Label("job", "x") not in change.labels
    assert change.skipped is True


def test_run_labels_add_a_job_pair_within_the_cap() -> None:
    change = full_set(62).add_run(pairs("job=x", "a=1"))
    assert change.applied == (Label("a", "1"), Label("job", "x"))
    assert change.skipped is False


def test_run_labels_refuse_non_job_pairs_beyond_the_cap() -> None:
    with pytest.raises(GrammarError, match="64"):
        full_set().add_run(pairs("a=1", "job=x"))


def test_run_labels_existing_job_pair_is_a_noop_not_a_skip() -> None:
    start = full_set(63).add(pairs("job=x")).labels
    change = start.add_run(pairs("job=x"))
    assert change.changed is False and change.skipped is False


def test_parse_run_labels_caps_only_non_job_pairs() -> None:
    texts = [f"k={index}" for index in range(64)]
    parsed = labels.parse_run_labels([*texts, "job=x", "job=y"])
    assert parsed[-2:] == [Label("job", "x"), Label("job", "y")]
    with pytest.raises(GrammarError, match="64"):
        labels.parse_run_labels([*texts, "k=64"])
    with pytest.raises(GrammarError, match="reserved"):
        labels.parse_run_labels(["kind=x"])


def test_parse_user_labels_caps_job_pairs_like_any_pair() -> None:
    texts = [f"k={index}" for index in range(64)]
    with pytest.raises(GrammarError, match="64"):
        labels.parse_user_labels([*texts, "job=x"])
