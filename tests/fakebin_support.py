"""Helpers shared by the fake-binary tests."""

from __future__ import annotations

import json
from typing import Any

from tests import sibling_schemas
from tests.fakebin.harness import Fakebin


def run_json(
    fakebin: Fakebin, tool: str, *args: str, schema: str | None = None, **kwargs: Any
) -> tuple[int, Any]:
    """Run a fake with ``--json``, parse stdout and (optionally) validate it against ``schema``."""
    split = args.index("--") if "--" in args else len(args)  # --json must precede any "--"
    done = fakebin.run(tool, *args[:split], "--json", *args[split:], **kwargs)
    assert done.stdout.strip(), (
        f"{tool} {args} printed nothing (exit {done.returncode}): {done.stderr}"
    )
    body = json.loads(done.stdout)
    if schema is not None:
        sibling_schemas.validate(body, tool, schema)
    return done.returncode, body


def error_of(fakebin: Fakebin, tool: str, *args: str, **kwargs: Any) -> tuple[int, dict[str, Any]]:
    """Run a call expected to fail; the stdout must be the tool's error envelope."""
    code, body = run_json(fakebin, tool, *args, schema="cli/error.json", **kwargs)
    return code, body["error"]
