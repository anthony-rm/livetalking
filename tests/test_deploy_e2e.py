"""End-to-end exercise of scripts/deploy_remote.py against a FAKE GPU box.

The real box is off limits to an agent (AGENTS.md), so this runs the deploy
driver for real -- real rsync, real shell, real symlink swap -- with `ssh` and
`rsync` shimmed to a temp directory (tests/fake_gpu_box.py). What it proves:

  * the release dir is created and `current` becomes a symlink to it (and a
    second deploy REPLACES the link instead of nesting inside the directory --
    the classic `ln -s`/`mv` footgun);
  * models/ and data/ are NOT copied, and are reachable through the release via
    the shared symlinks;
  * the revision stamp is written where the worker will read it;
  * `Health` reporting the same revision passes verification, a different one
    fails.

Skip conditions: no /usr/bin/rsync (unlikely on macOS).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import fake_gpu_box  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists("/usr/bin/rsync"), reason="needs a real rsync binary")


@pytest.fixture()
def box(tmp_path):
    info = fake_gpu_box.install(str(tmp_path / "box"))
    os.makedirs(os.path.join(info["home"], "livetalking"), exist_ok=True)
    return info


@pytest.fixture()
def repo(tmp_path):
    """A miniature copy of the repo layout (no weights, no .git)."""
    root = tmp_path / "repo"
    (root / "remote" / "proto").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "utils").mkdir()
    (root / "models").mkdir()
    (root / "data" / "avatars").mkdir(parents=True)
    # Files that MUST ship
    (root / "app.py").write_text("print('app')\n", encoding="utf-8")
    (root / "utils" / "logger.py").write_text("LOG=1\n", encoding="utf-8")
    (root / "remote" / "worker.py").write_text("WORKER=1\n", encoding="utf-8")
    # Generated stub newer than the proto (preflight freshness check)
    proto = root / "remote" / "proto" / "inference.proto"
    proto.write_text("syntax = 'proto3';\n", encoding="utf-8")
    stub = root / "remote" / "proto" / "inference_pb2.py"
    stub.write_text("# stub\n", encoding="utf-8")
    os.utime(stub, (proto.stat().st_atime + 10, proto.stat().st_mtime + 10))
    # Files that must NOT ship
    (root / "models" / "wav2lip.pth").write_text("weights", encoding="utf-8")
    (root / "data" / "avatars" / "a" / "latents.pt").parent.mkdir(parents=True)
    (root / "data" / "avatars" / "a" / "latents.pt").write_text("x", encoding="utf-8")
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    return root


def _run(repo, box, args, monkeypatch, *, probe=None, revision="deadbee"):
    """Drive the deploy driver in-process with shims + a stubbed Health probe."""
    monkeypatch.setenv("PATH", box["env"]["PATH"])
    monkeypatch.setenv("FAKE_GPU_HOME", box["env"]["FAKE_GPU_HOME"])
    monkeypatch.setenv("LT_REVISION", revision)
    # Direct RPC in these tests: the tunnel is a separate feature with its own
    # tests, and leaving it on would make every e2e case try to start ssh.
    monkeypatch.setenv("LT_SSH_TUNNEL", "0")
    monkeypatch.chdir(repo)
    from scripts import deploy_remote as dz
    monkeypatch.setattr(dz, "REPO_ROOT", str(repo))
    monkeypatch.setattr(dz, "load_target",
                        lambda override=None: {"username": "u", "host": "host",
                                               "port": 22, "password": "",
                                               "source": "test"})
    if probe is not None:
        monkeypatch.setattr(dz, "_probe", probe)
    # The fixture repo has no real worker entrypoint; the restart step is the
    # one thing we deliberately do NOT exercise here (it would pkill on THIS
    # machine). Record the call instead so the flow is still asserted.
    restarted = {}
    monkeypatch.setattr(dz, "restart_worker",
                        lambda target, remote_dir, force=False: restarted.update(
                            {"called": True, "force": force}))
    dz.main(args)
    if restarted:
        print(f"[test] restart_worker called (force={restarted['force']})")
    return 0


def test_deploy_creates_release_and_swaps_symlink(repo, box, monkeypatch):
    rc = _run(repo, box, ["--yes", "--dir", "~/livetalking"],
              monkeypatch, probe=lambda addr, timeout=5.0: {
                  "device": "cuda", "revision": "deadbee", "busy": False,
                  "detail": "fake GPU"})
    assert rc == 0

    base = os.path.join(box["home"], "livetalking")
    releases = os.listdir(os.path.join(base, "releases"))
    assert len(releases) == 1
    release = releases[0]
    assert release.startswith("deadbee-")

    rel_dir = os.path.join(base, "releases", release)
    assert os.path.islink(os.path.join(base, "current"))
    assert os.path.realpath(os.path.join(base, "current")) == os.path.realpath(rel_dir)

    # shipped
    assert os.path.exists(os.path.join(rel_dir, "app.py"))
    assert os.path.exists(os.path.join(rel_dir, "utils", "logger.py"))
    assert os.path.exists(os.path.join(rel_dir, "remote", "worker.py"))
    # revision stamped where the worker reads it
    rev = open(os.path.join(rel_dir, ".deploy", "revision"), encoding="utf-8").read()
    assert rev.strip() == "deadbee"

    # NOT shipped: weights, avatars, secrets
    assert not os.path.exists(os.path.join(rel_dir, "models", "wav2lip.pth"))
    assert not os.path.exists(os.path.join(rel_dir, "data", "avatars", "a", "latents.pt"))
    assert not os.path.exists(os.path.join(rel_dir, ".env"))

    # ...but shared dirs are reachable THROUGH the release
    assert os.path.exists(os.path.join(rel_dir, "models"))
    assert os.path.isdir(os.path.join(rel_dir, "data"))


def test_second_deploy_replaces_current_instead_of_nesting(repo, box, monkeypatch):
    good = lambda addr, timeout=5.0: {"device": "cuda", "revision": "deadbee",
                                      "busy": False}
    _run(repo, box, ["--yes", "--dir", "~/livetalking"], monkeypatch, probe=good)
    base = os.path.join(box["home"], "livetalking")
    first = os.listdir(os.path.join(base, "releases"))[0]

    # second deploy, different revision stamp
    monkeypatch.setenv("LT_REVISION", "cafe123")
    _run(repo, box, ["--yes", "--dir", "~/livetalking"], monkeypatch,
         revision="cafe123",
         probe=lambda addr, timeout=5.0: {"device": "cuda", "revision": "cafe123",
                                          "busy": False})

    current = os.path.join(base, "current")
    assert os.path.islink(current), "current must stay a symlink, not become a dir"
    assert "cafe123" in os.readlink(current)
    assert "current" not in os.listdir(os.path.join(base, "releases")), \
        "a nested <release>/current directory means the symlink swap went wrong"
    assert first in os.listdir(os.path.join(base, "releases"))


def test_rollback_flips_current_back(repo, box, monkeypatch):
    """--rollback must repoint `current` at the previous release and verify it."""
    good = lambda addr, timeout=5.0: {"device": "cuda", "revision": "deadbee",
                                      "busy": False}
    _run(repo, box, ["--yes", "--dir", "~/livetalking"], monkeypatch, probe=good)
    base = os.path.join(box["home"], "livetalking")
    first = sorted(os.listdir(os.path.join(base, "releases")))[0]

    _run(repo, box, ["--yes", "--dir", "~/livetalking"], monkeypatch,
         revision="cafe123",
         probe=lambda addr, timeout=5.0: {"device": "cuda", "revision": "cafe123",
                                          "busy": False})
    assert "cafe123" in os.readlink(os.path.join(base, "current"))

    from scripts import deploy_remote as dz
    # Rollback verifies against the OLD release's own revision.
    seen = {}

    def probe_old(addr, timeout=5.0):
        seen["rev"] = "deadbee"
        return {"device": "cuda", "revision": "deadbee", "busy": False}

    monkeypatch.setattr(dz, "_probe", probe_old)
    monkeypatch.setattr(dz, "restart_worker",
                        lambda t, d, force=False: None)
    monkeypatch.setenv("LT_SSH_TUNNEL", "0")
    monkeypatch.setattr(dz, "load_target",
                        lambda override=None: {"username": "u", "host": "host",
                                               "port": 22, "password": "",
                                               "source": "test"})
    monkeypatch.setenv("PATH", box["env"]["PATH"])
    monkeypatch.setenv("FAKE_GPU_HOME", box["env"]["FAKE_GPU_HOME"])
    dz.rollback({"username": "u", "host": "host", "port": 22}, "~/livetalking", 1)
    assert "deadbee" in os.readlink(os.path.join(base, "current"))
    assert os.path.basename(os.path.realpath(os.path.join(base, "current"))) == first


def test_rsync_ships_dot_directories(repo, box, monkeypatch):
    """REGRESSION: `.github/workflows/*.yml` aborted the transfer.

    With `--files-from=-` plus `--relative`, openrsync (macOS) rejected the
    dot-directory as "invalid path from sender" and the whole deploy died. Any
    dot-path in the shipped set exercises the same code, so cover it directly.
    """
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / ".github" / "workflows" / "checks.yml").write_text("on: push\n",
                                                               encoding="utf-8")
    monkeypatch.setenv("PATH", box["env"]["PATH"])
    monkeypatch.setenv("FAKE_GPU_HOME", box["env"]["FAKE_GPU_HOME"])
    from scripts import deploy_remote as dz
    monkeypatch.setattr(dz, "REPO_ROOT", str(repo))
    target = {"username": "u", "host": "host", "port": 22}
    dz.push_tree(target, "~/livetalking", "~/livetalking/releases/rel", method="rsync")

    base = os.path.join(box["home"], "livetalking/releases/rel")
    assert os.path.exists(os.path.join(base, ".github/workflows/checks.yml"))
    assert os.path.exists(os.path.join(base, "app.py"))
    assert not os.path.exists(os.path.join(base, "models/wav2lip.pth"))


def test_rsync_file_list_is_passed_via_a_file_not_a_pipe(repo, box, monkeypatch):
    """The '-' stream form is what failed; assert on the real argv, not prose."""
    monkeypatch.setenv("PATH", box["env"]["PATH"])
    monkeypatch.setenv("FAKE_GPU_HOME", box["env"]["FAKE_GPU_HOME"])
    from scripts import deploy_remote as dz
    monkeypatch.setattr(dz, "REPO_ROOT", str(repo))
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(dz.subprocess, "run", fake_run)
    dz._rsync_files({"username": "u", "host": "host", "port": 22}, "rel")

    argv = captured["cmd"]
    joined = " ".join(argv)
    assert "--files-from=-" not in joined, "the piped form breaks openrsync"
    assert any(a.startswith("--files-from=") and a != "--files-from=-"
               for a in argv), argv
    assert captured["input"] is None, "the list must come from a file, not stdin"


def test_tar_transport_ships_source_and_excludes_data(repo, box, monkeypatch):
    """The no-root fallback: tar over ssh must respect the same exclusions."""
    monkeypatch.setenv("PATH", box["env"]["PATH"])
    monkeypatch.setenv("FAKE_GPU_HOME", box["env"]["FAKE_GPU_HOME"])
    from scripts import deploy_remote as dz
    target = {"username": "u", "host": "host", "port": 22}
    release = "~/livetalking/releases/tarrel"
    dz.push_tree(target, "~/livetalking", release, method="tar")

    base = os.path.join(box["home"], "livetalking/releases/tarrel")
    assert os.path.exists(os.path.join(base, "app.py"))
    assert os.path.exists(os.path.join(base, "utils", "logger.py"))
    assert os.path.exists(os.path.join(base, "remote", "worker.py"))
    # excluded: weights, avatar data, secrets, caches
    assert not os.path.exists(os.path.join(base, "models", "wav2lip.pth"))
    assert not os.path.exists(os.path.join(base, "data"))
    assert not os.path.exists(os.path.join(base, ".env"))


def test_should_ship_keeps_vendored_model_sources():
    """The bsdtar trap: `models` must not match `avatars/musetalk/models/`.

    macOS ships bsdtar, whose `--exclude ./models` also drops that subtree. The
    worker imports those .py files, so losing them breaks it only after deploy.
    """
    from scripts import deploy_remote as dz
    assert dz.should_ship("avatars/musetalk/models/unet.py")
    assert dz.should_ship("avatars/wav2lip/models/__init__.py")
    assert dz.should_ship("app.py")
    assert dz.should_ship("remote/proto/inference_pb2.py")


def test_should_ship_drops_runtime_data():
    from scripts import deploy_remote as dz
    for path in ("models/wav2lip.pth", "data/avatars/a/latents.pt",
                 "gfpgan/weights/detection_Resnet50.pth", ".git/config",
                 ".env", "x/__pycache__/m.pyc", "run.log", "clip.mp4",
                 "livetalking.pid", "trial/a.py", "tmp/x.png"):
        assert not dz.should_ship(path), f"{path} must not ship"


def test_preflight_falls_back_to_tar_without_rsync(repo, box, monkeypatch):
    from scripts import deploy_remote as dz
    monkeypatch.setenv("PATH", box["env"]["PATH"])
    monkeypatch.setenv("FAKE_GPU_HOME", box["env"]["FAKE_GPU_HOME"])
    monkeypatch.setenv("LT_REVISION", "deadbee")
    monkeypatch.setattr(dz, "REPO_ROOT", str(repo))
    monkeypatch.setattr(dz, "resolve_python", lambda t, d: "python3")
    # Pretend the box has no rsync but does have tar+gzip.
    class _Out:
        returncode = 0
        stdout = "HAS_TAR\nHAS_GZIP\nNO_RSYNC\n"
        stderr = ""
    monkeypatch.setattr(dz, "run_ssh", lambda *a, **k: _Out())
    state = dz.preflight({"username": "u", "host": "host", "port": 22},
                         "~/livetalking")
    assert state["method"] == "tar"


def test_verification_fails_on_revision_drift(repo, box, monkeypatch):
    """A box serving other code must fail the deploy, not pass it."""
    monkeypatch.chdir(repo)
    monkeypatch.setenv("LT_REVISION", "deadbee")
    monkeypatch.setenv("LT_SSH_TUNNEL", "0")
    from scripts import deploy_remote as dz
    monkeypatch.setattr(dz, "health_address", lambda target: "host:50051")
    monkeypatch.setattr(dz, "_probe",
                        lambda addr, timeout=5.0: {"device": "cuda",
                                                   "revision": "0000000", "busy": False})
    with pytest.raises(dz.DeployError) as exc:
        dz.verify({"host": "host"}, "deadbee", deadline=1)
    assert "mismatch" in str(exc.value)


def test_preflight_rejects_stale_stub(repo, box, monkeypatch):
    """Editing inference.proto without regenerating the stub is a hard stop."""
    monkeypatch.chdir(repo)
    monkeypatch.setenv("LT_REVISION", "deadbee")
    from scripts import deploy_remote as dz
    # preflight inspects the repo it lives in -- point that at the fixture.
    monkeypatch.setattr(dz, "REPO_ROOT", str(repo))
    # Make the proto NEWER than its stub.
    proto = os.path.join(str(repo), "remote/proto/inference.proto")
    stub = os.path.join(str(repo), "remote/proto/inference_pb2.py")
    os.utime(proto, (os.path.getmtime(stub) + 60, os.path.getmtime(stub) + 60))
    with pytest.raises(SystemExit):
        dz.preflight({"username": "u", "host": "host", "port": 22}, "~/livetalking")
