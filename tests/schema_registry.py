"""A ``referencing`` registry over every schema in ``schemas/``, keyed by ``$id``.

Schemas use ids such as ``agent-spawn/cli/error.json`` and relative ``$ref``s such as
``../defs.json#/$defs/agent``, so every file must be in one registry to validate.
Golden-file tests call :func:`validate` with the ``$id`` of the schema they check.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"


def schema_files() -> list[Path]:
    return sorted(SCHEMAS_DIR.rglob("*.json"))


@cache
def _documents() -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for path in schema_files():
        document = json.loads(path.read_text(encoding="utf-8"))
        schema_id = document["$id"]
        if schema_id in documents:
            raise ValueError(f"duplicate $id {schema_id!r} ({path})")
        documents[schema_id] = document
    return documents


def schema_ids() -> list[str]:
    return sorted(_documents())


def schema(schema_id: str) -> dict[str, Any]:
    try:
        return _documents()[schema_id]
    except KeyError:
        raise KeyError(f"no schema with $id {schema_id!r}; known: {schema_ids()}") from None


@cache
def registry() -> Registry[Any]:
    resources = [
        (schema_id, Resource.from_contents(document, default_specification=DRAFT202012))
        for schema_id, document in _documents().items()
    ]
    return Registry().with_resources(resources)  # pyright: ignore[reportUnknownMemberType]


def validator(schema_id: str) -> Draft202012Validator:
    return Draft202012Validator(
        schema(schema_id), registry=registry(), format_checker=FormatChecker()
    )


def errors(instance: Any, schema_id: str) -> Iterator[str]:
    for error in validator(schema_id).iter_errors(instance):
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        yield f"{location}: {error.message}"


def validate(instance: Any, schema_id: str) -> None:
    """Raise ``AssertionError`` listing every violation of ``schema_id``."""
    problems = list(errors(instance, schema_id))
    if problems:
        raise AssertionError(f"{schema_id} rejected the instance:\n" + "\n".join(problems))
