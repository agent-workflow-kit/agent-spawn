"""The command table: one handler per command key.

To implement a command, replace the stub body in its group module; this table needs no edit
unless a command moves to a new module. Tests replace entries with ``monkeypatch.setitem``.
"""

from __future__ import annotations

from . import ending, internal, launching, maintenance, observing
from .base import Handler

HANDLERS: dict[str, Handler] = {
    "launch": launching.launch,
    "resume": launching.resume,
    "wait": observing.wait,
    "list": observing.list_agents,
    "show": observing.show,
    "attach": observing.attach,
    "ack": observing.ack,
    "label add": observing.label_add,
    "label remove": observing.label_remove,
    "stop": ending.stop,
    "teardown": ending.teardown,
    "prune": ending.prune,
    "reconcile": maintenance.reconcile,
    "cron install": maintenance.cron_install,
    "cron uninstall": maintenance.cron_uninstall,
    "export": maintenance.export,
    "browse": maintenance.browse,
    "supervise": internal.supervise,
    "_session-event": internal.session_event,
}
