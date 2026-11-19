"""``python -m tests.live``: list the live scenarios (``make live-list``), starting nothing."""

from __future__ import annotations

import os
import sys

from . import harness
from .scenarios import format_listing, parse_selection


def main() -> int:
    try:
        names = parse_selection(os.environ.get(harness.SCENARIOS_ENV))
        cap = harness.max_sessions(os.environ)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_listing(cap, names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
