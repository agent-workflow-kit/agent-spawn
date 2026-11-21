"""agent-spawn."""

from __future__ import annotations


def __getattr__(name: str) -> str:
    # The release job owns the version in pyproject.toml; the installed package metadata is the
    # only reader-facing copy. Looked up lazily so the hook relay does not pay for it.
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("agent-spawn")
        except PackageNotFoundError:  # only when run from an uninstalled source tree
            return "0.0.0"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
