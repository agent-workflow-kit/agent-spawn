"""``agent-spawn browse`` as a process: missing Textual, no terminal, the intent file, nesting."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests import schema_registry
from tests.browse_support import PtyRun
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.inspect_support import A1, A2, seed_agent
from tests.launch_support import Env

RESULT = "agent-spawn/cli/browse-result.json"
ERROR = "agent-spawn/cli/error.json"
NO_TEXTUAL = (
    "import sys;sys.modules['textual']=None;"
    "from agent_spawn.cli import main;raise SystemExit(main(sys.argv[1:]))"
)


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def browse_argv(*args: str) -> list[str]:
    return [sys.executable, "-m", "agent_spawn", "browse", *args]


def test_without_textual_it_exits_2_with_the_install_hint(env: Env, tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    done = subprocess.run(
        [sys.executable, "-c", NO_TEXTUAL, "--json", "browse", "--result-file", str(path)],
        capture_output=True,
        text=True,
        env=env.env(),
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert done.returncode == 2
    body = json.loads(done.stdout)
    schema_registry.validate(body, ERROR)
    assert body["error"]["code"] == "usage"
    assert "agent-spawn[tui]" in body["error"]["message"]
    assert "agent-spawn[tui]" in done.stderr
    assert not path.exists()  # browse writes nothing unless it ran


def test_without_a_terminal_it_exits_2_and_writes_nothing(env: Env, tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    done = subprocess.run(
        browse_argv("--result-file", str(path)),
        capture_output=True,
        text=True,
        env=env.env(),
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert done.returncode == 2 and "terminal" in done.stderr
    assert not path.exists()
    assert not env.state_dir.exists()


def run_browse(env: Env, *args: str) -> PtyRun:
    return PtyRun(browse_argv(*args), env.env())


def read_intent(path: Path) -> dict[str, object]:
    body = json.loads(path.read_text())
    schema_registry.validate(body, RESULT)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    return body


def test_it_lists_real_agents_and_q_without_a_result_file_writes_nothing(
    env: Env, tmp_path: Path
) -> None:
    seed_agent(env, A1, name="first one")
    seed_agent(env, A2, name="second [one]")
    before = sorted(p.name for p in tmp_path.iterdir())
    run = run_browse(env)
    run.read_until(A1)
    run.read_until("second [one]")  # markup brackets shown literally
    run.send("q")
    assert run.finish() == 0
    assert "\x1b[?1049l" in run.buffer  # the alternate screen is left again
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_q_with_a_result_file_records_back(env: Env, tmp_path: Path) -> None:
    seed_agent(env, A1)
    path = tmp_path / "out.json"
    run = run_browse(env, "--result-file", str(path))
    run.read_until(A1)
    run.send("q")
    assert run.finish() == 0
    assert read_intent(path) == {"intent": "back", "selected": []}


def test_attach_and_resume_record_intents_and_exit(env: Env, tmp_path: Path) -> None:
    seed_agent(env, A1)
    seed_agent(env, A2, created=1.0)  # older: listed second
    path = tmp_path / "out.json"
    run = run_browse(env, "--result-file", str(path))
    run.read_until(A2)
    run.send("a")
    assert run.finish() == 0
    assert read_intent(path) == {"intent": "attach", "selected": [A1]}
    path.unlink()
    run = run_browse(env, "--result-file", str(path))
    run.read_until(A2)
    run.send(" ")  # mark A1
    run.send("\x1b[B")  # down
    run.send(" ")  # mark A2
    run.send("r")
    assert run.finish() == 0
    assert read_intent(path) == {"intent": "resume", "selected": [A1, A2]}


def test_focus_starts_on_that_agent(env: Env, tmp_path: Path) -> None:
    seed_agent(env, A1, name="top")
    seed_agent(env, A2, name="target agent", created=1.0)
    path = tmp_path / "out.json"
    run = run_browse(env, "--focus", A2, "--result-file", str(path))
    run.read_until("target agent")
    run.read_until(f"agent: {A2}")  # the detail pane shows the focused agent
    run.send("a")
    assert run.finish() == 0
    assert read_intent(path) == {"intent": "attach", "selected": [A2]}


def test_a_bad_result_file_path_exits_2_before_the_screen(env: Env) -> None:
    run = run_browse(env, "--result-file", "")
    assert run.finish() == 2


def test_it_runs_as_a_foreground_child_of_another_tui(env: Env, tmp_path: Path) -> None:
    seed_agent(env, A1)
    path = tmp_path / "out.json"
    parent = textwrap.dedent(
        f"""
        import subprocess, sys
        sys.stdout.write("\\x1b[?1049h\\x1b[2JPARENT-UP\\n"); sys.stdout.flush()
        code = subprocess.run({browse_argv("--result-file", str(path))!r}).returncode
        sys.stdout.write("PARENT-BACK-%d\\n" % code); sys.stdout.flush()
        """
    )
    run = PtyRun([sys.executable, "-c", parent], env.env())
    run.read_until("PARENT-UP")
    run.read_until(A1)
    run.send("q")
    run.read_until("PARENT-BACK-0")
    assert run.finish() == 0
    assert read_intent(path) == {"intent": "back", "selected": []}


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_sigterm_and_sighup_leave_the_screen_cleanly_and_record_back(
    env: Env, tmp_path: Path, sig: signal.Signals
) -> None:
    seed_agent(env, A1)
    path = tmp_path / "out.json"
    run = run_browse(env, "--result-file", str(path))
    run.read_until(A1)
    os.kill(run.proc.pid, sig)  # the exact child this test started
    assert run.finish() == 0
    assert "\x1b[?1049l" in run.buffer  # the alternate screen is left again
    assert read_intent(path) == {"intent": "back", "selected": []}


FAKE_ATTACH_APP = textwrap.dedent(
    """
    import sys
    from agent_spawn.tui.app import BrowseApp
    from agent_spawn.tui.model import CliResult, SubprocessBackend

    ROW = {"agent_id": "ag-aaaaaaaa", "status": "live", "name": "the agent", "labels": []}

    class Backend(SubprocessBackend):
        def call(self, command, *, acting=False, timeout=None, terminal=False):
            if command[0] == "list":
                return CliResult(0, {"agents": [ROW]})
            if command[0] == "attach":
                self.terminal_flag = terminal
                return CliResult(0, {"action": "attach", "argv": ["sh", "-c", "exit 7"]})
            return CliResult(1, None, "unexpected")

    BrowseApp(Backend(config_path=None, actor="human:test")).run()
    """
)


def test_attach_suspends_the_screen_and_reports_the_exit_of_the_attach_argv(
    env: Env, tmp_path: Path
) -> None:
    script = tmp_path / "app.py"
    script.write_text(FAKE_ATTACH_APP)
    run = PtyRun([sys.executable, "-I", str(script)], env.env())
    run.read_until("the agent")
    run.send("a")
    run.read_until("back (exit 7)")
    run.send("q")
    assert run.finish() == 0


def test_a_terminal_call_gives_the_child_this_terminals_stdin_and_stderr(
    env: Env, tmp_path: Path
) -> None:
    script = tmp_path / "probe.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            from agent_spawn.tui.model import SubprocessBackend

            code = (
                "import json,sys;"
                "print(json.dumps({'in': sys.stdin.isatty(), 'out': sys.stdout.isatty(),"
                " 'err': sys.stderr.isatty()}))"
            )
            backend = SubprocessBackend(
                config_path=None, actor="human:test", prefix=[sys.executable, "-I", "-c", code]
            )
            print("TERMINAL", backend.call(["attach"], terminal=True).payload)
            print("PLAIN", backend.call(["attach"]).payload)
            """
        )
    )
    run = PtyRun([sys.executable, "-I", str(script)], env.env())
    run.read_until("PLAIN")
    assert run.finish() == 0
    assert "TERMINAL {'in': True, 'out': False, 'err': True}" in run.buffer
    assert "PLAIN {'in': False, 'out': False, 'err': False}" in run.buffer
