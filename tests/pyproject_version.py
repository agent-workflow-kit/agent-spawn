"""The version the release job writes to ``pyproject.toml``, read fresh (never a copy)."""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def pyproject_version() -> str:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
