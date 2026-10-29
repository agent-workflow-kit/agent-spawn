"""Single boundary for subprocess (and, later, HTTP) calls made by agent-spawn.

Every external call goes through this module so tests can substitute fakes
(see design guide §14). Commands are run as argv lists, never through a shell.
"""

from __future__ import annotations

import subprocess


def run(
    argv: list[str],
    *,
    stdin: str | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` without a shell, capturing text stdout and stderr."""
    return subprocess.run(
        argv,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
