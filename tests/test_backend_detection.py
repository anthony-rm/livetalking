#!/usr/bin/env python3
"""Tests for artifact-based backend selection on connect.

The bug these pin: the server used to pick its avatar backend once at startup
(--model / --remote_backend) and never revisit it, so connecting an avatar whose
artifacts belong to a DIFFERENT backend died on a missing file -- e.g. a wav2lip
avatar (face_imgs/, no latents.pt) against a musetalk-configured server:

    RuntimeError: Avatar 'anthony009' looks like a wav2lip avatar ...

Now the backend is derived from each avatar's artifacts. These tests cover the
detector and the connect-path dispatch. Inference is always local: there is no
remote backend to switch to any more.

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


def _ditto_avatar(root, avatar_id, *, complete=True, marker=True, face=False):
    """A ditto artifact: ditto.json + f_s/ + x_s_info/ + full_imgs/ + coords.pkl.

    `complete=False` omits one payload dir so the partial-artifact path is
    exercised; `face=True` adds face_imgs/ to reproduce the wav2lip collision.
    """
    path = _avatar(root, avatar_id, face=face)
    if marker:
        (path / "ditto.json").write_text('{"version": 1, "frame_count": 3}')
    (path / "full_imgs").mkdir(exist_ok=True)
    (path / "full_imgs" / "00000000.png").write_bytes(b"full")
    if complete:
        (path / "f_s").mkdir()
        (path / "f_s" / "00000000.npy").write_bytes(b"fs")
        (path / "x_s_info").mkdir()
        (path / "x_s_info" / "00000000.npz").write_bytes(b"xs")
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

    fake_opt = types.SimpleNamespace(
        model="musetalk", avatar_id="anthony007",
        batch_size=8, fps=25, transport="webrtc", tts="edgedts",
        sessionid="s")
    monkeypatch.setattr(app_mod, "opt", fake_opt)
    # Point detection at the fixture root by patching the module constant the
    # detector reads.
    monkeypatch.setattr(detect, "DEFAULT_AVATAR_ROOT", str(tmp_path))
    return app_mod, fake_opt, tmp_path


def test_resolves_each_avatar_to_its_own_backend(app_module):
    """One server fronts a mixed data/avatars/ dir; the artifacts decide."""
    app_mod, _opt, root = app_module
    _avatar(root, "ms", latents=True, mask=True)
    _avatar(root, "w2l", face=True)
    _avatar(root, "ul", face=True, ultralight=True)
    assert app_mod.resolve_avatar_backend("ms") == "musetalk"
    assert app_mod.resolve_avatar_backend("w2l") == "wav2lip"
    assert app_mod.resolve_avatar_backend("ul") == "ultralight"


def test_configured_model_does_not_override_the_artifacts(app_module):
    """A musetalk-configured server must still load a wav2lip avatar."""
    app_mod, opt, root = app_module
    opt.model = "musetalk"
    _avatar(root, "w2l", face=True)
    assert app_mod.resolve_avatar_backend("w2l") == "wav2lip"


def test_unrecognizable_avatar_is_rejected(app_module):
    app_mod, _opt, root = app_module
    (root / "junk").mkdir()
    with pytest.raises(RuntimeError, match="cannot load avatar"):
        app_mod.resolve_avatar_backend("junk")


def test_inference_is_never_dispatched_remotely(app_module):
    """The worker split is gone; nothing may report remote any more."""
    app_mod, _opt, _root = app_module
    assert app_mod.worker_remote() is False


def test_default_backend_is_a_real_local_backend(app_module):
    """A stale `model: remote` must not survive as a backend name."""
    app_mod, opt, _root = app_module
    opt.model = "wav2lip"
    assert app_mod.default_backend() == "wav2lip"


# ─── ditto ────────────────────────────────────────────────────────────────
def test_detects_ditto_from_its_marker(tmp_path):
    _ditto_avatar(tmp_path, "dt")
    backend, reason = detect.detect_backend("dt", root=str(tmp_path))
    assert backend == "ditto"
    assert "ditto.json" in reason


def test_ditto_wins_over_wav2lip_when_face_imgs_are_present(tmp_path):
    """The collision this rule exists for.

    A ditto tree can carry face_imgs/ as well; without checking ditto.json
    first the directory would detect as wav2lip and then fail to load.
    """
    _ditto_avatar(tmp_path, "dt", face=True)
    assert detect.detect_backend("dt", root=str(tmp_path))[0] == "ditto"


def test_partial_ditto_artifact_is_named_not_guessed(tmp_path):
    """A missing payload must report ditto + what is missing, not wav2lip."""
    _ditto_avatar(tmp_path, "dt", complete=False)
    backend, reason = detect.detect_backend("dt", root=str(tmp_path))
    assert backend == "ditto"
    assert "incomplete" in reason
    assert "f_s" in reason or "x_s_info" in reason


def test_ultralight_still_wins_over_ditto_less_trees(tmp_path):
    """Regression guard: the ditto rule must not shadow ultrafast layouts."""
    _avatar(tmp_path, "ul", face=True, ultralight=True)
    assert detect.detect_backend("ul", root=str(tmp_path))[0] == "ultralight"


def test_ditto_is_a_known_backend():
    assert "ditto" in detect.KNOWN_BACKENDS


def test_no_marker_means_no_ditto(tmp_path):
    """Without ditto.json the tree falls through to the legacy rules."""
    _ditto_avatar(tmp_path, "dt", marker=False, face=True)
    assert detect.detect_backend("dt", root=str(tmp_path))[0] == "wav2lip"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
