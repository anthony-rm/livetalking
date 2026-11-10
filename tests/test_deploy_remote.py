"""Unit tests for the deploy tooling + revision identity.

These never touch the network and never touch the GPU box: they exercise the
pure logic that decides *what* a deploy does (which paths ship, how a remote
path is quoted, how a revision mismatch is judged). The parts that must run
against the real box are only ever run by the user, per AGENTS.md.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remote import revision  # noqa: E402
from scripts import deploy_remote as dz  # noqa: E402


# ─── revision identity ─────────────────────────────────────────────────────
def test_matches_ignores_dirty_marker():
    assert revision.matches("abc1234+", "abc1234")
    assert revision.matches("abc1234", "abc1234+")
    assert revision.matches("abc1234", "abc1234")


def test_matches_rejects_different_or_empty():
    assert not revision.matches("abc1234", "def5678")
    assert not revision.matches("", "abc1234")
    assert not revision.matches("abc1234", "")


def test_current_revision_prefers_env(monkeypatch):
    monkeypatch.setenv("LT_REVISION", "envrev")
    assert revision.current_revision() == "envrev"


def test_current_revision_reads_file(monkeypatch, tmp_path):
    monkeypatch.delenv("LT_REVISION", raising=False)
    path = tmp_path / "revision"
    path.write_text("filerev\n", encoding="utf-8")
    monkeypatch.setattr(revision, "file_revision", lambda p=revision.REVISION_FILE: "filerev")
    assert revision.current_revision() == "filerev"


def test_write_and_read_revision_roundtrip(tmp_path):
    target = tmp_path / ".deploy" / "revision"
    revision.write_revision("abc1234+", str(target))
    assert target.read_text(encoding="utf-8").strip() == "abc1234+"
    assert revision.file_revision(str(target)) == "abc1234+"


# ─── remote path quoting ───────────────────────────────────────────────────
def test_rq_expands_home_so_the_shell_sees_it():
    # A literal "~/x" would never be expanded by the remote shell.
    assert dz.rq("~/livetalking") == '"$HOME"/livetalking'
    assert dz.rq("~") == '"$HOME"'


def test_rq_quotes_hostile_paths():
    assert dz.rq("/tmp/a b") == "'/tmp/a b'"
    assert dz.rq("/tmp/$(rm -rf /)") == "'/tmp/$(rm -rf /)'"


# ─── exclude rules ─────────────────────────────────────────────────────────
def test_excludes_cover_weights_data_and_env():
    """One predicate governs both transports, so weight/data/secrets cannot slip."""
    from scripts import deploy_remote as dz
    for must in ("models/wav2lip.pth", "data/avatars/a/latents.pt", ".env",
                 ".git/config", "gfpgan/weights/x.pth"):
        assert not dz.should_ship(must), f"{must} must never be deployed"


def test_excludes_do_not_swallow_package_sources():
    """A basename-only rule would also drop the vendored avatars/*/models/*.py."""
    from scripts import deploy_remote as dz
    assert dz.should_ship("avatars/musetalk/models/unet.py")
    assert dz.should_ship("avatars/wav2lip/models/__init__.py")


# ─── ssh plumbing (mocked) ─────────────────────────────────────────────────
class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_ssh_reports_failure_with_output(monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return _Proc(returncode=2, stderr="rsync: command not found")

    monkeypatch.setattr(dz.subprocess, "run", fake_run)
    target = {"username": "u", "host": "h", "port": 22, "password": "", "source": "test"}
    with pytest.raises(SystemExit):
        dz.run_ssh(target, "true")


def test_health_address_falls_back_to_ssh_host(monkeypatch):
    monkeypatch.setattr("remote.settings.effective_address", lambda: "")
    assert dz.health_address({"host": "10.0.0.9"}) == "10.0.0.9:50051"


def test_health_address_prefers_effective_setting(monkeypatch):
    monkeypatch.setattr("remote.settings.effective_address", lambda: "1.2.3.4:6000")
    assert dz.health_address({"host": "10.0.0.9"}) == "1.2.3.4:6000"


# ─── verification logic ────────────────────────────────────────────────────
def test_verify_rejects_missing_revision(monkeypatch):
    monkeypatch.setenv("LT_SSH_TUNNEL", "0")   # direct mode: no tunnel to open
    monkeypatch.setattr("remote.settings.effective_address", lambda: "h:1")
    monkeypatch.setattr(dz, "_probe",
                        lambda addr, timeout=5.0: {"device": "cuda", "revision": ""})
    with pytest.raises(dz.DeployError) as exc:
        dz.verify({"host": "h"}, "abc1234", deadline=1)
    assert "OLD code" in str(exc.value)


def test_verify_rejects_revision_drift(monkeypatch):
    monkeypatch.setenv("LT_SSH_TUNNEL", "0")
    monkeypatch.setattr("remote.settings.effective_address", lambda: "h:1")
    monkeypatch.setattr(dz, "_probe",
                        lambda addr, timeout=5.0: {"device": "cuda", "revision": "old0000"})
    with pytest.raises(dz.DeployError) as exc:
        dz.verify({"host": "h"}, "abc1234", deadline=1)
    assert "mismatch" in str(exc.value)


def test_verify_rejects_non_cuda(monkeypatch):
    monkeypatch.setenv("LT_SSH_TUNNEL", "0")
    monkeypatch.setattr("remote.settings.effective_address", lambda: "h:1")
    monkeypatch.setattr(dz, "_probe",
                        lambda addr, timeout=5.0: {"device": "cpu", "revision": "abc1234"})
    with pytest.raises(dz.DeployError) as exc:
        dz.verify({"host": "h"}, "abc1234", deadline=1)
    assert "cuda" in str(exc.value)


def test_verify_accepts_matching_cuda_worker(monkeypatch):
    monkeypatch.setenv("LT_SSH_TUNNEL", "0")
    monkeypatch.setattr("remote.settings.effective_address", lambda: "h:1")
    monkeypatch.setattr(dz, "_probe",
                        lambda addr, timeout=5.0: {"device": "cuda", "revision": "abc1234+",
                                                  "detail": "NVIDIA A100"})
    info = dz.verify({"host": "h"}, "abc1234", deadline=1)
    assert info["device"] == "cuda"


# ─── exit-code transport and the self-kill regression ──────────────────────
class TestExitCodeTransport:
    """ssh exits 255 for auth failure AND for a remote command that exits 255.

    Treating every 255 as "no usable key" turned a self-inflicted kill into a
    bogus auth error, sending the user to --setup-key for a working key.
    """

    def test_marker_survives_a_remote_exit_255(self):
        out, rc = dz._split_rc("some output\n__LT_RC=255\n")
        assert rc == 255
        assert out == "some output"

    def test_remote_nonzero_is_not_blamed_on_the_key(self, monkeypatch):
        seen = {}

        def fake_run(cmd, **kw):
            seen["argv"] = cmd
            return _Proc(returncode=255, stdout="RESTARTED=DEAD\n__LT_RC=4\n")

        monkeypatch.setattr(dz.subprocess, "run", fake_run)
        proc = dz.run_ssh({"username": "u", "host": "h", "port": 22},
                          "exit 4", check=False)
        assert proc.returncode == 4, "remote exit code was lost"
        assert any("__LT_RC=$?" in a for a in seen["argv"])

    def test_no_marker_and_255_is_reported_as_transport(self, monkeypatch):
        monkeypatch.setattr(dz.subprocess, "run", lambda cmd, **kw: _Proc(
            returncode=255, stderr="Permission denied (publickey)."))
        captured = {}
        monkeypatch.setattr(dz, "fail", lambda msg: captured.setdefault("msg", msg))
        dz.run_ssh({"username": "u", "host": "h", "port": 22}, "true", check=False)
        msg = captured.get("msg", "")
        assert "transport" in msg
        assert "Permission denied" in msg, "the real ssh error must be shown"

    def test_marker_is_stripped_from_stdout(self, monkeypatch):
        monkeypatch.setattr(dz.subprocess, "run", lambda cmd, **kw: _Proc(
            returncode=0, stdout="current -> rel\n__LT_RC=0\n"))
        proc = dz.run_ssh({"username": "u", "host": "h", "port": 22}, "true")
        assert "__LT_RC" not in proc.stdout
        assert proc.stdout == "current -> rel"


class TestRestartDoesNotKillItself:
    """REGRESSION: `pkill -f remote.worker` matched the restart shell itself.

    The script text contains the literal "remote.worker", so when the whole
    script was passed as an ssh command argument it became part of that bash
    process's own cmdline. pkill killed the session, ssh returned 255, and the
    deploy blamed the SSH key.
    """

    def test_restart_runs_from_a_file_not_an_ssh_argument(self):
        import inspect
        src = inspect.getsource(dz.restart_worker)
        assert "run_script(" in src, "restart must not pass its text as an ssh arg"
        assert "run_ssh(target, script" not in src

    def test_kill_loop_excludes_own_pid_and_ancestors(self):
        import inspect
        src = inspect.getsource(dz.restart_worker)
        assert "OURS=" in src, "must exclude its own pid from the kill set"
        assert "PPID" in src, "must exclude its own parent chain"

    def test_script_body_never_appears_in_the_ssh_argv(self, monkeypatch):
        """The root cause, asserted directly.

        `pkill -f`/`pgrep -f` match the whole cmdline. Passing the script as an
        ssh argument put its text -- including the literal "remote.worker" --
        into that bash process's own argv, so the kill loop matched and TERMed
        its own shell. run_script must ship the body out-of-band (base64 to a
        temp file) and invoke only `bash /tmp/...`.
        """
        import base64
        seen = []

        def fake_run(cmd, **kw):
            seen.append(list(cmd))
            return _Proc(returncode=0, stdout="__LT_RC=0\n")

        monkeypatch.setattr(dz.subprocess, "run", fake_run)
        body = "for pid in $(pgrep -f remote[.]worker); do kill $pid; done\n"
        dz.run_script({"username": "u", "host": "h", "port": 22}, body,
                      label="restart")

        assert len(seen) >= 2, "expected a write step and a run step"
        # The write step carries the body only as base64 (which cannot contain
        # the literal pattern), never as plain shell source.
        blob = seen[0][-1]
        assert "remote[.]worker" not in " ".join(seen[0])
        assert base64.b64encode(body.encode()).decode()[:24] in blob

        # The run step invokes a path, so no pattern can match its cmdline.
        run_argv = seen[-1]
        assert "remote[.]worker" not in " ".join(run_argv)
        joined = " ".join(run_argv)
        assert "bash /tmp/.lt-deploy-restart-" in joined, run_argv
        assert "\n" not in joined, "the script body must not be an argv element"


# ─── restart diagnostics ───────────────────────────────────────────────────
class TestRestartDiagnostics:
    """`RESTARTED=DEAD` plus a stale log tail diagnoses nothing."""

    def test_restart_preflights_with_check_before_daemonising(self):
        """--check must run BEFORE nohup, and from the release directory.

        `-m remote.worker` only resolves from current/, and a missing import
        must be reported as itself rather than as a dead pid two seconds later.
        """
        import linecache
        linecache.checkcache()
        with open(dz.__file__, encoding="utf-8") as fh:
            src = fh.read()
        body = src[src.index("def restart_worker"):]
        body = body[:body.index("def _probe")]
        # Compare within the shell script only: the docstring mentions nohup.
        script = body[body.index('script = f"""'):]
        # Self-contained -c snippet, NOT a project module: --restart-only may
        # target a release predating any helper we add locally.
        assert "remote.preflight" not in script
        assert 'PREFLIGHT_PY=' in script
        assert script.index("PREFLIGHT_PY") < script.index("nohup")
        assert 'cd "$BASE/current" || exit 6' in body

    def test_log_tail_starts_at_this_runs_offset(self):
        """The log is append-only; a bare tail -30 can show a PREVIOUS worker."""
        import linecache
        linecache.checkcache()
        with open(dz.__file__, encoding="utf-8") as fh:
            src = fh.read()
        body = src[src.index("def restart_worker"):]
        body = body[:body.index("def _probe")]
        assert "LOG_MARK=" in body
        assert 'tail -c +$((LOG_MARK + 1))' in body

    def test_preflight_failure_explains_the_dependency_fix(self, monkeypatch):
        monkeypatch.setattr(dz, "run_script", lambda *a, **k: _Proc(
            returncode=5, stdout="PREFLIGHT=FAILED\n  grpc: No module named grpc\n"))
        monkeypatch.setattr(dz, "wait_for_idle", lambda *a, **k: None)
        monkeypatch.setattr(dz, "resolve_python", lambda *a, **k: "/env/bin/python")
        captured = {}
        def fake_fail(msg):
            captured.setdefault("msg", msg)
            raise dz.DeployError(msg)

        monkeypatch.setattr(dz, "fail", fake_fail)
        with pytest.raises(dz.DeployError):
            dz.restart_worker({"host": "h"}, "~/livetalking")
        msg = captured["msg"]
        assert "no process was started" in msg, "must say nothing is running"
        assert "pip install grpcio opencv-python-headless tqdm" in msg
        assert "Do NOT install -r requirements.txt" in msg
        assert "/env/bin/python" in msg


class TestPreflightCoversRealImports:
    """A preflight that checks fewer modules than the worker imports is a lie.

    grpcio >= 1.8x does not bundle protobuf, and inference_pb2 imports
    `google.protobuf` directly. Checking only "grpc" let a worker pass the
    preflight and die during import.
    """

    def _preflight(self):
        """The executable snippet only -- NOT the comments around it.

        Asserting against the whole region is a trap: a comment mentioning
        google.protobuf made the first version of this test pass even with the
        module removed from the tuple.
        """
        import linecache
        linecache.checkcache()
        with open(dz.__file__, encoding="utf-8") as fh:
            src = fh.read()
        start = src.index("PREFLIGHT_PY='")
        body = src[start:src.index("if ! OUT=", start)]
        return "\n".join(line for line in body.splitlines()
                         if not line.lstrip().startswith("#"))

    def _modules(self):
        import re
        m = re.search(r"for mod in \(([^)]*)\)", self._preflight())
        assert m, "preflight module tuple not found"
        return {tok.strip().strip('"\'') for tok in m.group(1).split(",")}

    def test_preflight_checks_google_protobuf(self):
        assert "google.protobuf" in self._modules(), (
            "protobuf is not bundled with grpcio; omitting it reproduces the "
            "worker dying on `from google.protobuf import descriptor`")

    def test_preflight_is_self_contained(self):
        """No project imports: --restart-only may target an older release."""
        body = self._preflight()
        assert "remote.preflight" not in body
        assert "import remote" not in body
        assert "remote.worker" not in body

    def test_preflight_modules_match_the_workers_imports(self):
        """Every third-party module worker.py needs must be in the preflight."""
        modules = self._modules()
        root = os.path.dirname(os.path.dirname(os.path.abspath(dz.__file__)))
        needed = set()
        for rel in ("remote/worker.py", "remote/tensors.py", "utils/image.py"):
            with open(os.path.join(root, rel), encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith(("import ", "from ")) and "remote" not in line:
                        mod = line.split()[1].split(".")[0]
                        if mod in ("grpc", "numpy", "cv2", "tqdm", "google", "torch"):
                            needed.add(mod)
        assert needed, "sanity: the scan must find imports"
        for mod in needed:
            key = "google.protobuf" if mod == "google" else mod
            assert key in modules, (
                f"{key} is imported by the worker but not preflighted")
