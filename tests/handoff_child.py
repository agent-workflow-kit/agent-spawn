"""The window-process stand-in for the handoff process tests.

Run as ``python handoff_child.py <socket> <timeout> <result file>``: it calls
``receive_environment`` with its own environment as the pane environment and writes the outcome
(``{"env": ...}`` or ``{"failed": <cause>}``) to the result file, atomically. An optional
fourth argument is a marker file touched once the imports are done, just before connecting.
"""

from __future__ import annotations

import json
import os
import sys

from agent_spawn.handoff import HandoffFailed, receive_environment


def main(argv: list[str]) -> int:
    socket_path, timeout, result = argv[1], float(argv[2]), argv[3]
    if len(argv) > 4:
        open(argv[4], "w").close()  # started: imports are done
    before = dict(os.environ)
    try:
        env = receive_environment(socket_path, timeout=timeout, pane_env=before)
        outcome: dict[str, object] = {"env": env, "before": before, "pid": os.getpid()}
        code = 0
    except HandoffFailed as exc:
        outcome = {"failed": exc.cause, "before": before, "pid": os.getpid()}
        code = 1
    with open(result + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(outcome, handle)
    os.rename(result + ".tmp", result)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
