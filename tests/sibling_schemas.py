"""Validate against the siblings' schemas, copied under ``tests/sibling-schemas/``.

Each sibling's schemas are copied (never imported or read from the sibling's checkout) into
``tests/sibling-schemas/<tool>/<version>/`` and pinned to the interface version agent-spawn is
built against. A ``PIN.json`` next to them records the tool, the interface version and the
sibling commit the files were copied from. Schemas use ``$id`` values such as
``termctl/cli/open.json`` with relative ``$ref``s, so each tool gets its own registry.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

ROOT = Path(__file__).resolve().parent / "sibling-schemas"
TOOLS = ("claude-session", "termctl", "wtctl")
PINNED_VERSION = "0.1.0-design"


def pin_dir(tool: str, version: str = PINNED_VERSION) -> Path:
    return ROOT / tool / version


@cache
def _documents(tool: str, version: str) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for path in sorted(pin_dir(tool, version).rglob("*.json")):
        if path.name == "PIN.json":
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        schema_id = document["$id"]
        if schema_id in documents:
            raise ValueError(f"duplicate $id {schema_id!r} ({path})")
        documents[schema_id] = document
    return documents


def schema_ids(tool: str, version: str = PINNED_VERSION) -> list[str]:
    return sorted(_documents(tool, version))


@cache
def _registry(tool: str, version: str) -> Registry[Any]:
    resources = [
        (schema_id, Resource.from_contents(document, default_specification=DRAFT202012))
        for schema_id, document in _documents(tool, version).items()
    ]
    return Registry().with_resources(resources)  # pyright: ignore[reportUnknownMemberType]


def errors(instance: Any, tool: str, schema_name: str, version: str = PINNED_VERSION) -> list[str]:
    """Violations of ``<tool>/<schema_name>`` (e.g. ``cli/open.json``); empty when valid."""
    schema_id = f"{tool}/{schema_name}"
    documents = _documents(tool, version)
    if schema_id not in documents:
        raise KeyError(f"no schema {schema_id!r} in {pin_dir(tool, version)}")
    validator = Draft202012Validator(
        documents[schema_id], registry=_registry(tool, version), format_checker=FormatChecker()
    )
    found: list[str] = []
    for error in validator.iter_errors(instance):
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        found.append(f"{location}: {error.message}")
    return found


def validate(instance: Any, tool: str, schema_name: str, version: str = PINNED_VERSION) -> None:
    """Raise ``AssertionError`` listing every violation of the pinned sibling schema."""
    problems = errors(instance, tool, schema_name, version)
    if problems:
        raise AssertionError(
            f"{tool}/{schema_name} ({version}) rejected the instance:\n" + "\n".join(problems)
        )
