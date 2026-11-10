"""Tests for remote avatar creation (GenerateAvatar RPC + host dispatch).

The real flow needs a GPU box, which this agent must not drive. These tests pin
the parts that decide correctness without one:

* the proto exposes the RPC and the generated stubs match it;
* the host's task manager routes generation to the worker when an address is
  configured, and unpacks the returned zip into data/avatars/<id>/ atomically;
* a missing/unreachable worker does not silently fall back to a local run that
  would take hours on this machine.

Everything is stubbed at the client boundary — no network, no GPU.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from remote.proto import inference_pb2, inference_pb2_grpc  # noqa: E402


# ─── protocol surface ──────────────────────────────────────────────────────
def test_proto_declares_generate_avatar():
    """The RPC must exist in the generated stubs, not just the .proto."""
    assert "GenerateAvatar" in inference_pb2.DESCRIPTOR.services_by_name[
        "RemoteWorker"].methods_by_name
    assert hasattr(inference_pb2, "GenerateAvatarRequest")
    assert hasattr(inference_pb2, "GenerateAvatarReply")


def test_client_stub_exposes_generate_avatar():
    """gRPC binds methods per-instance, so assert on an instance, not the class."""
    import grpc
    stub = inference_pb2_grpc.RemoteWorkerStub(grpc.insecure_channel("127.0.0.1:1"))
    assert hasattr(stub, "GenerateAvatar")


def test_reply_carries_an_archive_and_failure_message():
    ok = inference_pb2.GenerateAvatarReply(ok=True, archive=b"PK\x03\x04", frames=3)
    assert ok.archive.startswith(b"PK")
    bad = inference_pb2.GenerateAvatarReply(ok=False, msg="no face found")
    assert bad.ok is False and "face" in bad.msg


# ─── host dispatch ─────────────────────────────────────────────────────────
def _make_task(tmp_path, avatar_id="avatar_remote"):
    from server.task_manager import AvatarTask
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not-a-real-video")
    return AvatarTask(
        task_id="t1", model_type="musetalk", avatar_id=avatar_id,
        params={"video_path": str(clip), "save_path": str(tmp_path / "avatars"),
                "bbox_shift": 0})


def _artifact_zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def remote_env(monkeypatch):
    """Pretend the toggle is on remote and a worker address is configured."""
    from remote import settings
    monkeypatch.setattr(settings, "effective_address", lambda: "127.0.0.1:50051")
    monkeypatch.setattr(settings, "worker_mode", lambda: settings.MODE_REMOTE)
    return settings


def test_should_use_remote_only_when_the_toggle_says_remote(monkeypatch):
    """A configured address must NOT pull work off this machine by itself."""
    from remote import settings
    from server.task_manager import TaskManager
    monkeypatch.setattr(settings, "effective_address", lambda: "127.0.0.1:50051")
    monkeypatch.setattr(settings, "worker_mode", lambda: settings.MODE_LOCAL)
    assert TaskManager()._should_use_remote() is False

    monkeypatch.setattr(settings, "worker_mode", lambda: settings.MODE_REMOTE)
    assert TaskManager()._should_use_remote() is True


def test_should_not_use_remote_when_the_toggle_is_local(monkeypatch):
    from remote import settings
    from server.task_manager import TaskManager
    monkeypatch.setattr(settings, "effective_address", lambda: "")
    monkeypatch.setattr(settings, "worker_mode", lambda: settings.MODE_LOCAL)
    assert TaskManager()._should_use_remote() is False


def test_run_remote_unpacks_the_artifact_tree(tmp_path, remote_env, monkeypatch):
    """The zip the worker returns must land as data/avatars/<id>/ verbatim."""
    from server import task_manager as tm

    archive = _artifact_zip({
        "latents.pt": b"L",
        "coords.pkl": b"C",
        "mask/00000000.png": b"M",
        "full_imgs/00000000.png": b"F",
        "avator_info.json": b"{}",
    })
    seen = {}

    class FakeClient:
        def __init__(self, address, **kw):
            seen["address"] = address

        def close(self):
            pass

        def generate_avatar(self, backend, avatar_id, *, source_bytes=None,
                            filename="", params=None, **kw):
            seen.update(backend=backend, avatar_id=avatar_id,
                        source_bytes=source_bytes, filename=filename,
                        params=params)
            return archive

    monkeypatch.setattr("remote.client.RemoteClient", FakeClient)

    mgr = tm.TaskManager()
    task = _make_task(tmp_path)
    mgr._run_remote(task, lambda p: None)

    dest = tmp_path / "avatars" / "avatar_remote"
    assert dest.is_dir()
    assert (dest / "latents.pt").read_bytes() == b"L"
    assert (dest / "mask" / "00000000.png").read_bytes() == b"M"
    assert (dest / "full_imgs" / "00000000.png").read_bytes() == b"F"
    # The clip is shipped as BYTES: the box cannot see this filesystem.
    assert seen["source_bytes"] == b"not-a-real-video"
    assert seen["backend"] == "musetalk"
    assert seen["avatar_id"] == "avatar_remote"
    assert task.params["generated_on"] == "remote"


def test_run_remote_replaces_a_previous_artifact(tmp_path, remote_env, monkeypatch):
    """Re-generating an existing id must not merge with the old tree."""
    from server import task_manager as tm

    dest = tmp_path / "avatars" / "avatar_remote"
    dest.mkdir(parents=True)
    (dest / "stale_file_from_previous_run.bin").write_bytes(b"old")

    archive = _artifact_zip({"latents.pt": b"new"})

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def close(self): pass
        def generate_avatar(self, *a, **kw): return archive

    monkeypatch.setattr("remote.client.RemoteClient", FakeClient)
    task = _make_task(tmp_path)
    tm.TaskManager()._run_remote(task, lambda p: None)

    assert not (dest / "stale_file_from_previous_run.bin").exists()
    assert (dest / "latents.pt").read_bytes() == b"new"


def test_run_remote_leaves_no_partial_dir_on_bad_zip(tmp_path, remote_env, monkeypatch):
    """A corrupt archive must fail loudly and never leave a half-built avatar."""
    from server import task_manager as tm

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def close(self): pass
        def generate_avatar(self, *a, **kw): return b"not a zip"

    monkeypatch.setattr("remote.client.RemoteClient", FakeClient)
    task = _make_task(tmp_path)
    with pytest.raises(zipfile.BadZipFile):
        tm.TaskManager()._run_remote(task, lambda p: None)
    assert not (tmp_path / "avatars" / "avatar_remote").exists()


def test_failed_remote_generation_marks_the_task_failed(tmp_path, remote_env, monkeypatch):
    """End-to-end through _run_task: a worker error is a failed task, not a
    silent multi-hour local run."""
    from server import task_manager as tm

    class BoomClient:
        def __init__(self, *a, **kw): pass
        def close(self): pass
        def generate_avatar(self, *a, **kw):
            raise RuntimeError("worker said no")

    monkeypatch.setattr("remote.client.RemoteClient", BoomClient)
    mgr = tm.TaskManager()
    task = _make_task(tmp_path)
    mgr.tasks[task.task_id] = task
    mgr._run_task(task.task_id)

    assert task.status == "failed"
    assert "worker said no" in task.error_msg
