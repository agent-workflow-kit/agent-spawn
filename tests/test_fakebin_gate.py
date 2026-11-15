"""The fakes' ``gate`` (hold a call until a file exists) and ``clear_script``."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tests.fakebin.harness import Fakebin


def forget(fakebin: Fakebin) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [str(fakebin.bin_dir / "claude-session"), "forget", "cs-abcdefgh", "--json"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=fakebin.env(),
    )


def test_a_gated_call_waits_for_the_file_and_then_answers_normally(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    fakebin.seed_session("cs-abcdefgh")
    gate = tmp_path / "open"
    fakebin.script("claude-session", "forget", gate=str(gate))
    proc = forget(fakebin)
    try:
        fakebin.wait_for_gate(gate)
        assert proc.poll() is None  # held: nothing answered yet
        gate.write_text("")
        out, _err = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:  # our own child, started above
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0
    assert json.loads(out)["forgotten"][0]["csession_id"] == "cs-abcdefgh"


def test_clear_script_disables_rules_but_keeps_the_counts_of_later_ones(
    fakebin: Fakebin,
) -> None:
    fakebin.seed_session("cs-abcdefgh")
    fakebin.seed_session("cs-bbbbbbbb")
    fakebin.seed_session("cs-cccccccc")
    fakebin.script("claude-session", "status", merge={"state": "working"})
    fakebin.script("claude-session", "forget", times=1, error={"code": "refused", "message": "x"})
    fakebin.clear_script("claude-session", "status")
    first = fakebin.run("claude-session", "forget", "cs-abcdefgh", "--json")
    second = fakebin.run("claude-session", "forget", "cs-bbbbbbbb", "--json")
    assert (first.returncode, second.returncode) == (20, 0)  # the times=1 rule still counts
    status = fakebin.run("claude-session", "status", "cs-cccccccc", "--json")
    assert status.returncode == 0 and json.loads(status.stdout)["state"] != "working"
