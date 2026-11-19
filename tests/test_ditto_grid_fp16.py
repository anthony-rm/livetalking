"""T15 — the fp16 grid-sample lane must be a strict opt-in.

The hybrid's two 5-D grid samples sit between fp32-I/O TRT segments, so the
fp16 lane is only a dtype choice for the sampler itself: inputs cast down,
result cast back up. These tests pin the contract that makes the env safe to
ship OFF:

  * default (DITTO_GRID_FP16 unset) returns the fp32 result BIT-IDENTICALLY
    — no cast, no rounding, so the shipped lane cannot drift;
  * the opt-in lane returns fp32 output of the same shape whose error stays
    at the hybrid's own acceptance scale (2e-4 mean) on real-shaped tensors;
  * the flag is read by the instance, not the class, so one process can run
    both arms (which the box gate relies on).
"""
import importlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture()
def hybrid_mod(monkeypatch):
    import avatars.ditto.trt_warp_hybrid as m
    monkeypatch.delenv("DITTO_GRID_FP16", raising=False)
    return importlib.reload(m)


def _bare(mod, fp16):
    """An instance without __init__ (no TRT engines on a CPU test host)."""
    obj = object.__new__(mod.TRTWarpHybrid)
    obj.grid_fp16 = fp16
    return obj


def _tensors(seed=7, n=2, c=4, d=6, h=16, w=16):
    g = torch.Generator().manual_seed(seed)
    vol = torch.randn(n, c, d, h, w, generator=g) * 8.0     # f_s-scale spread
    grid = torch.rand(n, d, h, w, 3, generator=g) * 2.0 - 1.0
    return vol, grid


def test_default_lane_is_bit_identical_to_fp32(hybrid_mod):
    vol, grid = _tensors()
    h = _bare(hybrid_mod, fp16=False)
    got = h._grid_sample_lane(vol, grid)
    ref = F.grid_sample(vol, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    assert got.dtype == torch.float32
    assert torch.equal(got, ref)


def test_env_flag_parses_off_by_default(hybrid_mod):
    assert hybrid_mod.GRID_FP16 is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
def test_env_flag_accepts_the_documented_on_values(monkeypatch, value):
    import avatars.ditto.trt_warp_hybrid as m
    monkeypatch.setenv("DITTO_GRID_FP16", value)
    assert importlib.reload(m).GRID_FP16 is True
    monkeypatch.delenv("DITTO_GRID_FP16")
    importlib.reload(m)


def test_fp16_lane_is_live_and_keeps_the_fp32_contract(hybrid_mod):
    """The opt-in lane must change output (it is not a silent no-op) while
    preserving the fp32/shape contract the TRT segments depend on.

    It does NOT assert the 2e-4 gate: on f_s-scale volumes the sampler's
    absolute error measured ~1.7e-2 on CPU, which is why T15 ships OFF and
    the box gate (full chain, real tensors) owns the verdict.
    """
    vol, grid = _tensors()
    h = _bare(hybrid_mod, fp16=True)
    got = h._grid_sample_lane(vol, grid)
    ref = F.grid_sample(vol, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    assert got.dtype == torch.float32, "engine segments expect fp32 input"
    assert got.shape == ref.shape
    assert not torch.equal(got, ref), "fp16 lane must not be a silent no-op"
    rel = ((got - ref).abs().mean() / ref.abs().mean()).item()
    assert rel < 0.05, f"fp16 sampler relative error {rel:.3f} is not rounding"


def test_flag_lives_on_the_instance_not_the_class(hybrid_mod):
    """The box gate runs both lanes in one process."""
    a = _bare(hybrid_mod, fp16=False)
    b = _bare(hybrid_mod, fp16=True)
    vol, grid = _tensors()
    ref = a._grid_sample_lane(vol, grid)
    other = b._grid_sample_lane(vol, grid)
    assert not torch.equal(ref, other)
    assert torch.equal(ref, a._grid_sample_lane(vol, grid))
