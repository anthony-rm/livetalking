#!/usr/bin/env python3
"""Idle-lane tests for the Ditto backend: a still image must not be frozen.

No weights, no GPU, no DITTO_HOME. The seam is `DittoReal.idle_frame`, which
`BaseAvatar.process_frames` calls in its silent branch; `_idle_motion` and the
loop/cache arithmetic are pure numpy + integer math, so the instance is built
with __new__ and the model objects are fakes that record what they were asked
to render.

The GPU worker is faked as a synchronous call: what these tests pin is the
CONTRACT (which frames differ, what is reused, that a video artifact opts out),
not the TensorRT plumbing.
"""
from __future__ import annotations

import numpy as np
import pytest

import avatars.ditto_avatar as da
from avatars.ditto_avatar import DittoReal, _X_S_SHAPES


# ─── fakes ─────────────────────────────────────────────────────────────────
class FakeStitch:
    """Passes x_d through and stamps it so tests can see which frame ran."""

    def __init__(self):
        self.d0 = None
        self.calls = []

    def __call__(self, x_s, x_d):
        self.calls.append(float(np.asarray(x_d["yaw"]).reshape(-1)[0]))
        return x_s, x_d


class FakeWarp:
    feature_edge = 8

    def __init__(self):
        self.calls = 0

    def __call__(self, f_s, x_s, x_d):
        self.calls += 1
        return np.zeros((256, 8, 8), np.float32)

    def run_batch_dev(self, chains):
        self.calls += len(chains)
        import torch
        return [torch.zeros((256, 8, 8)) for _ in chains]


class FakeDecode:
    """Batched decoder shape: (B,3,H,W) in 0..1, on the CPU."""

    batch = 10

    def __init__(self):
        import torch
        self._torch = torch

    def __call__(self, f_3d):
        t = f_3d if self._torch.is_tensor(f_3d) else self._torch.as_tensor(f_3d)
        n = t.shape[0]
        return self._torch.full((n, 3, 16, 16), 0.5)


def make_backend(n_frames=1, reuse=2, loop=8, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(da, "_IDLE_REUSE", reuse)
        monkeypatch.setattr(da, "_IDLE_LOOP", loop)
        monkeypatch.setattr(da, "_IDLE_MOTION", True)
    obj = object.__new__(DittoReal)
    obj.n_frames = n_frames
    obj.x_s_list = [{k: np.zeros(s, np.float32) for k, s in _X_S_SHAPES.items()}
                    for _ in range(n_frames)]
    obj.x_s_list[0]["yaw"] = np.full((1, 66), 0.5, np.float32)
    obj.frame_list_cycle = [np.zeros((32, 32, 3), np.uint8)
                            for _ in range(n_frames)]
    obj.M_c2o_list = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)]
    obj.marker = {"render_size": 16}
    obj._paste_geom_cache = {}
    obj._mask_cache = np.ones((16, 16, 3), np.float32)
    obj.stitch = FakeStitch()
    obj.warp = FakeWarp()
    obj.dec = FakeDecode()
    obj.infer_pipeline = None
    obj.render_edge = 8
    obj.f_s_list = [np.zeros((32, 8, 8, 16), np.float32)]
    obj._idle_tick = 0
    obj._idle_cache = {}
    obj._warp_pool_staged = lambda: False
    return obj


# ─── the hook contract ─────────────────────────────────────────────────────
def test_base_idle_frame_defaults_to_the_source_frame():
    """Every other backend keeps its old behaviour: the clip IS the idle."""
    from avatars.base_avatar import BaseAvatar
    obj = object.__new__(BaseAvatar)
    frame = np.full((4, 4, 3), 7, np.uint8)
    obj.frame_list_cycle = [frame]
    assert obj.idle_frame(0) is frame


def test_base_idle_frame_returns_none_without_frames():
    """A backend that never set frame_list_cycle must not raise in the loop."""
    from avatars.base_avatar import BaseAvatar
    assert object.__new__(BaseAvatar).idle_frame(0) is None


def test_video_artifact_opts_out_of_the_idle_lane(monkeypatch):
    """n_frames > 1 already moves (its own clip): rendering would be waste."""
    b = make_backend(n_frames=3, monkeypatch=monkeypatch)
    assert b.idle_frame(0) is None
    assert b._idle_cache == {}


def test_idle_motion_can_be_switched_off(monkeypatch):
    b = make_backend(monkeypatch=monkeypatch)
    monkeypatch.setattr(da, "_IDLE_MOTION", False)
    assert b.idle_frame(0) is None


# ─── motion synthesis ──────────────────────────────────────────────────────
def test_idle_motion_moves_the_pose_channels_not_the_expression(monkeypatch):
    """Breathing sways pose; exp/kp stay put (they are expression geometry)."""
    b = make_backend(monkeypatch=monkeypatch)
    a, c = b._idle_motion(0), b._idle_motion(da._IDLE_LOOP // 4)
    assert not np.allclose(a["yaw"], c["yaw"])
    assert not np.allclose(a["t"], c["t"])
    for key in ("exp", "kp"):
        assert np.array_equal(a[key], c[key])


def test_idle_motion_does_not_mutate_the_artifact(monkeypatch):
    b = make_backend(monkeypatch=monkeypatch)
    before = b.x_s_list[0]["yaw"].copy()
    b._idle_motion(3)
    assert np.array_equal(b.x_s_list[0]["yaw"], before)


def test_idle_motion_is_periodic_in_the_loop(monkeypatch):
    """A cached loop only repeats without a seam if index 0 == index LOOP."""
    b = make_backend(monkeypatch=monkeypatch)
    for i in (0, 1, 5):
        assert np.array_equal(b._idle_motion(i)["yaw"],
                              b._idle_motion(i + da._IDLE_LOOP)["yaw"])


# ─── loop + cache behaviour ────────────────────────────────────────────────
def test_first_idle_frame_renders_and_later_ticks_reuse_the_same_crop(monkeypatch):
    """Rendering is throttled: _IDLE_REUSE displayed frames per render."""
    b = make_backend(reuse=3, monkeypatch=monkeypatch)
    assert b.idle_frame(0) is not None
    assert b.warp.calls == 1
    # The next two displays are the SAME rendered frame — no GPU work.
    for _ in range(2):
        assert b.idle_frame(0) is not None
    assert b.warp.calls == 1


def test_the_loop_renders_each_index_once_then_stops_using_the_gpu(monkeypatch):
    b = make_backend(reuse=1, loop=4, monkeypatch=monkeypatch)
    for _ in range(4):                     # one full loop
        b.idle_frame(0)
    assert b.warp.calls == 4
    for _ in range(8):                     # a second pass: all cached
        b.idle_frame(0)
    assert b.warp.calls == 4
    assert len(b._idle_cache) == 4


def test_idle_frames_differ_across_the_loop(monkeypatch):
    """The point of the lane: consecutive idle frames are not the same picture."""
    seen = []

    class RecordingStitch(FakeStitch):
        def __call__(self, x_s, x_d):
            seen.append(float(np.asarray(x_d["yaw"]).reshape(-1)[0]))
            return x_s, x_d

    b = make_backend(reuse=1, loop=8, monkeypatch=monkeypatch)
    b.stitch = RecordingStitch()
    for _ in range(8):
        b.idle_frame(0)
    assert len(set(round(v, 6) for v in seen)) > 1


def test_idle_frame_returns_a_full_frame_matching_the_source(monkeypatch):
    b = make_backend(monkeypatch=monkeypatch)
    out = b.idle_frame(0)
    assert out is not None
    assert out.shape == b.frame_list_cycle[0].shape
    assert out.dtype == np.uint8


def test_a_render_failure_keeps_the_still_frame(monkeypatch):
    """A broken engine must degrade to the old behaviour, not kill the loop."""
    b = make_backend(monkeypatch=monkeypatch)

    def boom(*_a, **_k):
        raise RuntimeError("engine gone")

    b.warp = type("W", (), {"feature_edge": 8, "__call__": boom})()
    assert b.idle_frame(0) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
