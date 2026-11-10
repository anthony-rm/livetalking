#!/usr/bin/env python3
"""Tests for artifact-based backend selection on connect.

The bug these pin: the server used to pick its avatar backend once at startup
(--model / --remote_backend) and never revisit it, so connecting an avatar whose
artifacts belong to a DIFFERENT backend died on a missing file -- e.g. a wav2lip
avatar (face_imgs/, no latents.pt) against a musetalk-configured server:

    RuntimeError: Avatar 'anthony009' looks like a wav2lip avatar ...

Now the backend is derived from each avatar's artifacts. These tests cover the
detector, the connect-path dispatch, and the remote backend switching without a
GPU box (clients stubbed at the RPC boundary).

Run with the project interpreter:
    /opt/miniconda3/envs/livetalking/bin/python -m pytest tests/test_backend_detection.py
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avatars import detect  # noqa: E402


# ─── helpers ───────────────────────────────────────────────────────────────
def _avatar(root, avatar_id, *, latents=False, mask=False, face=False,
            ultralight=False, info=False):
    path = root / avatar_id
    path.mkdir(parents=True)
    (path / "coords.pkl").write_bytes(b"coords")
    if latents:
        (path / "latents.pt").write_bytes(b"latents")
    if mask:
        (path / "mask").mkdir()
        (path / "mask" / "00000000.png").write_bytes(b"m")
    if face:
        (path / "face_imgs").mkdir()
        (path / "face_imgs" / "00000000.png").write_bytes(b"f")
    if ultralight:
        (path / "ultralight.pth").write_bytes(b"w")
    if info:
        (path / "avator_info.json").write_text("{}")
    return path


# ─── detector ──────────────────────────────────────────────────────────────
def test_detects_musetalk_from_latents_and_mask(tmp_path):
    _avatar(tmp_path, "ms", latents=True, mask=True, info=True)
    assert detect.detect_backend("ms", root=str(tmp_path))[0] == "musetalk"


def test_detects_wav2lip_from_face_imgs(tmp_path):
    _avatar(tmp_path, "w2l", face=True)
    assert detect.detect_backend("w2l", root=str(tmp_path))[0] == "wav2lip"


def test_ultralight_wins_over_wav2lip_when_pth_present(tmp_path):
    """Both have face_imgs/; ultralight.pth is the discriminator."""
    _avatar(tmp_path, "ul", face=True, ultralight=True)
    assert detect.detect_backend("ul", root=str(tmp_path))[0] == "ultralight"


def test_latents_without_mask_is_still_musetalk(tmp_path):
    """Half-built musetalk: name the backend so its loader explains the gap."""
    _avatar(tmp_path, "half", latents=True)
    backend, reason = detect.detect_backend("half", root=str(tmp_path))
    assert backend == "musetalk" and "mask" in reason


def test_avator_info_alone_suggests_musetalk(tmp_path):
    _avatar(tmp_path, "legacy", info=True)
    assert detect.detect_backend("legacy", root=str(tmp_path))[0] == "musetalk"


def test_missing_dir_reports_why(tmp_path):
    backend, reason = detect.detect_backend("nope", root=str(tmp_path))
    assert backend is None and "does not exist" in reason


def test_unrecognized_layout_reports_what_was_expected(tmp_path):
    (tmp_path / "weird").mkdir()
    (tmp_path / "weird" / "random.bin").write_bytes(b"x")
    backend, reason = detect.detect_backend("weird", root=str(tmp_path))
    assert backend is None and "latents.pt" in reason


def test_real_repo_avatars_report_their_backend():
    """Sanity-check against the avatars actually checked out here."""
    expected = {
        "anthony005": "musetalk",
        "anthony007": "musetalk",
        "anthony009": "wav2lip",
        "wav2lip256_avatar1": "wav2lip",
    }
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "avatars")
    for avatar_id, backend in expected.items():
        if not os.path.isdir(os.path.join(root, avatar_id)):
            pytest.skip(f"{avatar_id} not present in this checkout")
        assert detect.detect_backend(avatar_id, root=root)[0] == backend


# ─── connect-path dispatch (app.resolve_avatar_backend) ────────────────────
@pytest.fixture
def app_module(monkeypatch, tmp_path):
    """Import app with a stub opt and a temporary avatar root."""
    import app as app_mod
    from remote import settings

    fake_opt = types.SimpleNamespace(
        model="remote", remote_backend="musetalk", avatar_id="anthony007",
        batch_size=8, fps=25, transport="webrtc", tts="edgedts",
        sessionid="s")
    monkeypatch.setattr(app_mod, "opt", fake_opt)
    # The local/remote toggle (not --model) decides where GPU work runs.
    monkeypatch.setattr(settings, "worker_mode", lambda: settings.MODE_REMOTE)
    # Point detection at the fixture root by patching the module constant the
    # detector reads.
    monkeypatch.setattr(detect, "DEFAULT_AVATAR_ROOT", str(tmp_path))
    return app_mod, fake_opt, tmp_path


def test_remote_host_stays_remote_plugin_for_worker_backends(app_module):
    app_mod, _opt, root = app_module
    _avatar(root, "ms", latents=True, mask=True)
    _avatar(root, "w2l", face=True)
    # In remote mode the HOST plugin is always 'remote'; the worker switches.
    assert app_mod.resolve_avatar_backend("ms") == "remote"
    assert app_mod.resolve_avatar_backend("w2l") == "remote"


def test_remote_falls_back_to_local_for_a_backend_the_worker_cannot_run(
        app_module, monkeypatch):
    """ultralight is host-only: remote mode must not fail the whole toggle."""
    app_mod, _opt, root = app_module
    _avatar(root, "ul", face=True, ultralight=True)
    assert app_mod.resolve_avatar_backend("ul") == "ultralight"


def test_local_mode_switches_to_the_avatars_own_backend(app_module, monkeypatch):
    from remote import settings
    app_mod, opt, root = app_module
    opt.model = "musetalk"          # configured for musetalk ...
    monkeypatch.setattr(settings, "worker_mode", lambda: settings.MODE_LOCAL)
    _avatar(root, "w2l", face=True)  # ... but this avatar is wav2lip
    assert app_mod.resolve_avatar_backend("w2l") == "wav2lip"


def test_local_mode_rejects_an_unrecognizable_avatar(app_module, monkeypatch):
    from remote import settings
    app_mod, opt, root = app_module
    opt.model = "musetalk"
    monkeypatch.setattr(settings, "worker_mode", lambda: settings.MODE_LOCAL)
    (root / "junk").mkdir()
    with pytest.raises(RuntimeError, match="cannot load avatar"):
        app_mod.resolve_avatar_backend("junk")


def test_local_toggle_overrides_a_stale_remote_model(app_module, monkeypatch):
    """`model: remote` in config.yaml must not outlive the toggle."""
    from remote import settings
    app_mod, opt, _root = app_module
    monkeypatch.setattr(settings, "worker_mode", lambda: settings.MODE_LOCAL)
    assert app_mod.default_backend() != "remote"


# ─── remote backend switching (avatars/remote_avatar.py) ───────────────────
class _FakeClient:
    """Records the RPC sequence so the switch can be asserted."""

    def __init__(self):
        self.calls = []
        self.loaded_backend = None

    def wait_ready(self, deadline=30.0):
        self.calls.append(("wait_ready",))
        return {"device": "cuda", "detail": "fake-gpu", "backend": ""}

    def load_model(self, backend):
        self.calls.append(("load_model", backend))
        self.loaded_backend = backend

    def load_avatar(self, backend, avatar_id, frames):
        self.calls.append(("load_avatar", backend, avatar_id))

    def unload(self):
        self.calls.append(("unload",))


@pytest.fixture
def remote_mod(monkeypatch):
    import avatars.remote_avatar as ra

    # Reset module state between tests.
    monkeypatch.setattr(ra, "_client", None)
    monkeypatch.setattr(ra, "_backend", "")
    monkeypatch.setattr(ra, "_fallback_backend", "musetalk")
    monkeypatch.setattr(ra, "_worker_avatar_id", "")
    monkeypatch.setattr(ra, "_avatar_backends", {})

    fake = _FakeClient()
    monkeypatch.setattr(ra, "RemoteClient", lambda address, **kw: fake)
    # A connect is never "live" in these unit tests.
    import server.session_manager as sm
    monkeypatch.setattr(sm.session_manager, "active_sessions", lambda: 0)
    return ra, fake


def test_load_model_connects_without_priming_weights(remote_mod):
    ra, fake = remote_mod
    ra.load_model(types.SimpleNamespace(remote_backend="musetalk"))
    # No LoadModel: the first avatar's artifacts decide the backend.
    assert ("load_model", "musetalk") not in fake.calls
    assert ("wait_ready",) in fake.calls


def test_switching_backend_reloads_worker_weights(remote_mod, monkeypatch):
    ra, fake = remote_mod
    monkeypatch.setattr(ra, "RemoteClient", lambda address, **kw: fake)
    ra.load_model(types.SimpleNamespace(remote_backend="musetalk"))

    ra._ensure_worker_backend("wav2lip")
    assert ("load_model", "wav2lip") in fake.calls
    assert ra._backend == "wav2lip"

    # Re-requesting the same backend must not reload.
    before = len(fake.calls)
    ra._ensure_worker_backend("wav2lip")
    assert len(fake.calls) == before


def test_switching_backend_is_refused_while_a_session_is_live(remote_mod, monkeypatch):
    ra, fake = remote_mod
    ra.load_model(types.SimpleNamespace(remote_backend="musetalk"))
    ra._ensure_worker_backend("musetalk")

    import server.session_manager as sm
    monkeypatch.setattr(sm.session_manager, "active_sessions", lambda: 1)
    with pytest.raises(RuntimeError, match="still connected"):
        ra._ensure_worker_backend("wav2lip")
    assert ra._backend == "musetalk"


def test_unknown_worker_backend_is_rejected(remote_mod):
    ra, _fake = remote_mod
    ra.load_model(types.SimpleNamespace(remote_backend="musetalk"))
    with pytest.raises(RuntimeError, match="cannot drive"):
        ra._ensure_worker_backend("ultralight")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
