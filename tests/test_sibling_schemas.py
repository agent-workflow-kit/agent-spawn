"""The copied sibling schemas are intact, pinned and self-consistent."""

from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from tests import sibling_schemas
from tests.sibling_schemas import PINNED_VERSION, TOOLS


@pytest.mark.parametrize("tool", TOOLS)
def test_each_tool_is_pinned_to_the_design_interface_version(tool: str) -> None:
    pin = json.loads((sibling_schemas.pin_dir(tool) / "PIN.json").read_text())
    assert pin["tool"] == tool
    assert pin["interface_version"] == PINNED_VERSION == "0.1.0-design"
    assert pin["source_repo_commit"]


@pytest.mark.parametrize("tool", TOOLS)
def test_every_schema_is_valid_and_every_ref_resolves(tool: str) -> None:
    ids = sibling_schemas.schema_ids(tool)
    assert ids, f"no schemas copied for {tool}"
    for schema_id in ids:
        document = sibling_schemas._documents(tool, PINNED_VERSION)[schema_id]  # pyright: ignore[reportPrivateUsage]
        Draft202012Validator.check_schema(document)
        assert schema_id.startswith(f"{tool}/")
        # Resolving every $ref (including the defs files) needs the whole tool's registry.
        registry = sibling_schemas._registry(tool, PINNED_VERSION)  # pyright: ignore[reportPrivateUsage]
        for ref in _refs(document):
            registry.resolver(base_uri=schema_id).lookup(ref)


def _refs(node: object) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():  # pyright: ignore[reportUnknownVariableType]
            if key == "$ref" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_refs(value))
    elif isinstance(node, list):
        for item in node:  # pyright: ignore[reportUnknownVariableType]
            found.extend(_refs(item))
    return found


def test_validation_rejects_a_bad_instance() -> None:
    problems = sibling_schemas.errors({"error": {"code": 1}}, "termctl", "cli/error.json")
    assert problems


def test_unknown_schema_is_an_error() -> None:
    with pytest.raises(KeyError):
        sibling_schemas.validate({}, "termctl", "cli/nope.json")
