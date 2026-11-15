"""Self-labeling (``label add --session``): every refusal, in order, and the happy path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import self_label
from agent_spawn.output import CliError
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.inspect_support import A1, A2, SESSION, labels_of, seed_agent
from tests.launch_support import Env

ADD = "agent-spawn/cli/label-add.json"
ERROR = "agent-spawn/cli/error.json"
SECRET = "s3cr3t-value"


class Rig:
    def __init__(
        self, env: Env, fakebin: Fakebin, iso: Isolation, *, scripted: bool = True
    ) -> None:
        self.env, self.fakebin = env, fakebin
        self.dir = iso.state / "claude-session" / "sessions" / SESSION
        self.dir.mkdir(parents=True)
        self.secret = self.dir / "secret"
        self.secret.write_text(SECRET + "\n")
        self.secret.chmod(0o600)
        fakebin.seed_session(SESSION)
        if scripted:
            fakebin.script("claude-session", "verify-secret", secret=SECRET)
        seed_agent(
            env,
            A1,
            csession_id=SESSION,
            claude_session_path=str(fakebin.bin_dir / "claude-session"),
        )

    def run(self, *pairs: str, session: str = SESSION, extra: dict[str, str] | None = None) -> Any:
        environ = {"CSESSION_ID": SESSION, "CSESSION_DIR": str(self.dir)}
        environ.update(extra or {})
        for name in [n for n, v in environ.items() if v == ""]:
            del environ[name]
        return self.env.run(
            "label", "add", "--session", session, *pairs, "--json", extra_env=environ
        )

    def labels(self) -> list[str]:
        return labels_of(self.env, A1)

    def verified(self) -> bool:
        return any(a[0] == "verify-secret" for a in self.fakebin.argvs("claude-session"))

    def refused(self, done: Any, code: int = 11) -> dict[str, Any]:
        body = json.loads(done.stdout)
        assert done.returncode == code, (done.stdout, done.stderr)
        schema_registry.validate(body, ERROR)
        assert SECRET not in done.stdout + done.stderr
        assert self.labels() == []
        return body


@pytest.fixture
def rig(fakebin: Fakebin, isolation_env: Isolation) -> Rig:
    return Rig(Env(fakebin, isolation_env), fakebin, isolation_env)


@pytest.fixture
def bare_rig(fakebin: Fakebin, isolation_env: Isolation) -> Rig:
    """A rig whose verify-secret answer the test scripts itself."""
    return Rig(Env(fakebin, isolation_env), fakebin, isolation_env, scripted=False)


def test_self_label_adds_and_mirrors(rig: Rig) -> None:
    done = rig.run("pr=h/o/r#7")
    body = json.loads(done.stdout)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, ADD)
    assert body["agent_id"] == A1 and body["added"] == ["pr=h/o/r#7"] and body["changed"]
    assert body["dry_run"] is False
    assert rig.labels() == ["pr=h/o/r#7"]
    calls = rig.fakebin.invocations("claude-session")
    verify = [c for c in calls if c.argv[0] == "verify-secret"]
    assert len(verify) == 1
    assert verify[0].argv == ["verify-secret", SESSION, "--json"]  # the secret is not in argv
    assert verify[0].stdin is not None and verify[0].stdin.strip() == SECRET
    assert SECRET not in json.dumps([c.argv for c in calls])
    assert any(c.argv[:3] == ["label", "add", SESSION] for c in calls)
    assert SECRET not in done.stdout + done.stderr


def test_self_label_again_is_a_no_op(rig: Rig) -> None:
    rig.run("pr=1")
    body = json.loads(rig.run("pr=1").stdout)
    assert body["changed"] is False and body["added"] == []


def test_the_id_must_equal_csession_id_in_full(rig: Rig) -> None:
    body = rig.refused(rig.run("pr=1", extra={"CSESSION_ID": "cs-zzzzzzzz"}), 2)
    assert body["error"]["code"] == "usage"
    rig.refused(rig.run("pr=1", extra={"CSESSION_ID": ""}), 2)
    rig.refused(rig.run("pr=1", session="cs-abcd"), 2)
    assert rig.fakebin.argvs("claude-session") == []


def test_csession_dir_must_be_set(rig: Rig) -> None:
    rig.refused(rig.run("pr=1", extra={"CSESSION_DIR": ""}), 2)
    assert rig.fakebin.argvs("claude-session") == []


def test_only_allowed_keys(rig: Rig) -> None:
    rig.refused(rig.run("owner=x"), 2)
    rig.refused(rig.run("pr=1", "owner=x"), 2)
    for reserved in ("agent=ag-aaaaaaaa", "kind=implement"):
        rig.refused(rig.run(reserved), 2)
    assert rig.fakebin.argvs("claude-session") == []


def test_config_widens_the_allowed_keys(rig: Rig, isolation_env: Isolation) -> None:
    cfg = isolation_env.config / "agent-spawn"
    cfg.mkdir(parents=True)
    (cfg / "config.yaml").write_text("self_label_keys: [pr, note]\n")
    assert rig.run("note=x").returncode == 0
    assert rig.run("owner=x").returncode == 2


def test_remove_is_not_available_with_session(rig: Rig) -> None:
    done = rig.env.run("label", "remove", "--session", SESSION, "pr=1", "--json")
    assert done.returncode == 2


def test_no_database_means_exit_12_and_nothing_is_created(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env)
    rig = Rig.__new__(Rig)
    rig.env, rig.fakebin = env, fakebin
    rig.dir = isolation_env.state / "claude-session" / "sessions" / SESSION
    rig.dir.mkdir(parents=True)
    done = rig.run("pr=1")
    body = json.loads(done.stdout)
    assert done.returncode == 12
    assert body["error"]["code"] == "failed"
    hint = body["error"]["details"]["hint"]
    assert "XDG_STATE_HOME" in hint and "XDG_CONFIG_HOME" in hint
    assert not env.state_dir.exists()
    assert fakebin.argvs("claude-session") == []


def test_tainted_session_is_refused_before_the_secret_is_read(rig: Rig) -> None:
    rig.fakebin.script("claude-session", "status", merge={"tainted": True})
    rig.refused(rig.run("pr=1"))
    assert not rig.verified()


def test_third_party_reading_run_is_refused(rig: Rig) -> None:
    rig.fakebin.script("claude-session", "status", merge={"run": {"reads_third_party": True}})
    rig.refused(rig.run("pr=1"))
    assert not rig.verified()


def test_a_tainted_agent_record_is_refused(rig: Rig) -> None:
    import sqlite3

    db = sqlite3.connect(rig.env.state_dir / "state.db")
    with db:
        db.execute("UPDATE agents SET tainted = 1")
    db.close()
    rig.refused(rig.run("pr=1"))


def test_status_failure_fails_closed(rig: Rig) -> None:
    rig.fakebin.script(
        "claude-session", "status", exit=12, error={"code": "failed", "message": "x"}
    )
    rig.refused(rig.run("pr=1"), 12)
    assert not rig.verified()


def test_unusable_status_fails_closed(rig: Rig) -> None:
    rig.fakebin.script("claude-session", "status", stdout={"nonsense": True})
    done = rig.run("pr=1")
    assert done.returncode != 0
    assert rig.labels() == [] and not rig.verified()


def test_csession_dir_must_match_the_session_dir(rig: Rig, isolation_env: Isolation) -> None:
    other = isolation_env.state / "elsewhere"
    other.mkdir()
    (other / "secret").write_text(SECRET)
    (other / "secret").chmod(0o600)
    rig.refused(rig.run("pr=1", extra={"CSESSION_DIR": str(other)}))
    assert not rig.verified()
    rig.refused(rig.run("pr=1", extra={"CSESSION_DIR": str(isolation_env.state / "nope")}))


def test_csession_dir_is_compared_by_realpath(rig: Rig, isolation_env: Isolation) -> None:
    link = isolation_env.state / "link"
    link.symlink_to(rig.dir)
    assert rig.run("pr=1", extra={"CSESSION_DIR": str(link)}).returncode == 0


def test_secret_must_not_be_a_symlink(rig: Rig, isolation_env: Isolation) -> None:
    real = isolation_env.state / "real-secret"
    real.write_text(SECRET)
    real.chmod(0o600)
    rig.secret.unlink()
    rig.secret.symlink_to(real)
    rig.refused(rig.run("pr=1"))
    assert not rig.verified()


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o400, 0o700, 0o666])
def test_secret_mode_must_be_exactly_0600(rig: Rig, mode: int) -> None:
    rig.secret.chmod(mode)
    rig.refused(rig.run("pr=1"))
    assert not rig.verified()


def test_secret_over_128_bytes_is_refused(rig: Rig) -> None:
    rig.secret.write_text("x" * 129)
    rig.refused(rig.run("pr=1"))
    assert not rig.verified()


def test_secret_must_be_a_regular_file(rig: Rig) -> None:
    rig.secret.unlink()
    rig.secret.mkdir()
    rig.refused(rig.run("pr=1"))
    rig.secret.rmdir()
    os.mkfifo(rig.secret, 0o600)  # a FIFO must not hang the open
    rig.refused(rig.run("pr=1"))
    assert not rig.verified()


def test_missing_and_empty_secret_are_refused(rig: Rig) -> None:
    rig.secret.write_text("")
    rig.refused(rig.run("pr=1"))
    rig.secret.unlink()
    rig.refused(rig.run("pr=1"))
    assert not rig.verified()


def test_secret_mismatch_is_exit_11(rig: Rig) -> None:
    rig.secret.write_text("wrong-secret")
    rig.refused(rig.run("pr=1"))
    assert rig.verified()


@pytest.mark.parametrize("answer", [{"match": False}, {"nothing": 1}, {"match": "true"}])
def test_verify_secret_must_say_match_true(bare_rig: Rig, answer: dict[str, Any]) -> None:
    bare_rig.fakebin.script("claude-session", "verify-secret", stdout=answer)
    bare_rig.refused(bare_rig.run("pr=1"))


def test_verify_secret_failure_is_passed_through_and_labels_nothing(bare_rig: Rig) -> None:
    bare_rig.fakebin.script(
        "claude-session", "verify-secret", exit=12, error={"code": "failed", "message": "x"}
    )
    bare_rig.refused(bare_rig.run("pr=1"), 12)


def test_no_agent_for_the_session_is_exit_10(rig: Rig) -> None:
    import sqlite3

    db = sqlite3.connect(rig.env.state_dir / "state.db")
    with db:
        db.execute("UPDATE agents SET csession_id = 'cs-zzzzzzzz'")
    db.close()
    rig.refused(rig.run("pr=1"), 10)


def test_a_torn_down_agent_is_not_labeled(rig: Rig) -> None:
    import sqlite3

    db = sqlite3.connect(rig.env.state_dir / "state.db")
    with db:
        db.execute("UPDATE agents SET status = 'torn_down', torn_down_at = 1")
    db.close()
    rig.refused(rig.run("pr=1"), 10)


def test_the_cap_applies_and_changes_nothing(rig: Rig) -> None:
    seed_agent(rig.env, A2)  # another agent is untouched by the failure below
    from agent_spawn.labels import LabelSet
    from agent_spawn.store import Store

    with Store.open(rig.env.state_dir) as store, store.lease(f"agent:{A1}") as lease:
        store.edit_labels(
            lease, A1, "add", list(LabelSet.from_texts([f"k{i}=v" for i in range(64)]).pairs)
        )
    done = rig.run("pr=1")
    assert done.returncode == 2
    assert len(rig.labels()) == 64 and "pr=1" not in rig.labels()


def test_mirror_failure_is_a_warning_not_a_failure(rig: Rig) -> None:
    rig.fakebin.script(
        "claude-session", "label add", exit=12, error={"code": "failed", "message": "m"}
    )
    body = json.loads(rig.run("pr=1").stdout)
    assert body["changed"] is True and len(body["warnings"]) == 1
    assert rig.labels() == ["pr=1"]


# --- read_secret directly ----------------------------------------------------------------------


def test_read_secret_checks_the_owner(tmp_path: Path) -> None:
    (tmp_path / "secret").write_text("abc")
    (tmp_path / "secret").chmod(0o600)
    assert self_label.read_secret(str(tmp_path), os.getuid()) == "abc"
    with pytest.raises(CliError) as caught:
        self_label.read_secret(str(tmp_path), os.getuid() + 1)
    assert caught.value.exit_code == 11


def test_read_secret_refuses_a_symlinked_directory_component_leaf(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "secret").write_text("abc")
    (real / "secret").chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(CliError):  # the directory itself is opened with O_NOFOLLOW
        self_label.read_secret(str(link), os.getuid())


def test_read_secret_refuses_non_utf8(tmp_path: Path) -> None:
    (tmp_path / "secret").write_bytes(b"\xff\xfe")
    (tmp_path / "secret").chmod(0o600)
    with pytest.raises(CliError):
        self_label.read_secret(str(tmp_path), os.getuid())
