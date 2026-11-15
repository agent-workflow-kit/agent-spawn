import pytest

from tests import schema_registry


def test_every_schema_file_is_a_valid_draft_2020_12_schema() -> None:
    from jsonschema import Draft202012Validator

    for schema_id in schema_registry.schema_ids():
        Draft202012Validator.check_schema(schema_registry.schema(schema_id))


def test_ids_are_unique_and_include_the_cli_and_hook_schemas() -> None:
    ids = schema_registry.schema_ids()
    assert len(ids) == len(schema_registry.schema_files())
    assert "agent-spawn/cli/error.json" in ids
    assert "agent-spawn/defs.json" in ids
    assert "agent-spawn/hooks/post-park.json" in ids


def test_relative_refs_resolve_across_files() -> None:
    bad = {"error": {"agent_id": "nope"}}
    problems = list(schema_registry.errors(bad, "agent-spawn/cli/error.json"))
    assert problems, "an invalid envelope must be rejected"
    # a $ref into defs.json is followed (no Unresolvable error) when agent_id is checked
    envelope = _minimal_error()
    envelope["error"]["agent_id"] = "ag-abcdefgh"
    schema_registry.validate(envelope, "agent-spawn/cli/error.json")
    envelope["error"]["agent_id"] = "ag-ABC"
    with pytest.raises(AssertionError, match="agent_id"):
        schema_registry.validate(envelope, "agent-spawn/cli/error.json")


def test_unknown_schema_id_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="agent-spawn/cli/error.json"):
        schema_registry.validate({}, "agent-spawn/cli/nope.json")


def _minimal_error() -> dict[str, dict[str, object]]:
    return {
        "error": {
            "code": "usage",
            "message": "m",
            "reason": None,
            "agent_id": None,
            "tool": "agent-spawn",
            "lower_code": None,
            "lower_error": None,
            "details": None,
            "candidates": [],
            "hook": None,
            "hook_stderr": None,
            "decision_ids": [],
        }
    }


@pytest.mark.parametrize("schema_id", ["agent-spawn/config.json", "agent-spawn/defs.json"])
@pytest.mark.parametrize("value", ["36500d", 3153600000, "1." + "0" * 30])
def test_duration_schema_accepts_the_grammar_bounds(schema_id: str, value: int | str) -> None:
    assert not _duration_errors(value, schema_id)


@pytest.mark.parametrize("schema_id", ["agent-spawn/config.json", "agent-spawn/defs.json"])
@pytest.mark.parametrize("value", ["1." + "0" * 31, "1." + "0" * 30 + "s", 3153600001])
def test_duration_schema_rejects_over_32_characters_or_over_36500_days(
    schema_id: str, value: int | str
) -> None:
    assert _duration_errors(value, schema_id)


def _duration_errors(value: int | str, schema_id: str) -> list[str]:
    if schema_id == "agent-spawn/config.json":
        return list(schema_registry.errors({"handoff_timeout": value}, schema_id))
    from jsonschema import Draft202012Validator

    ref = Draft202012Validator(
        {"$ref": f"{schema_id}#/$defs/duration"}, registry=schema_registry.registry()
    )
    return [error.message for error in ref.iter_errors(value)]
