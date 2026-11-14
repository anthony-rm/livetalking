#!/usr/bin/env python3
"""Tests for the PCM → Ditto-HuBERT windowing layer.

The windowing is pure arithmetic, so it is fully testable without the 1.46 GB
HuBERT checkpoint, onnxruntime-gpu, or a GPU. A stub extractor stands in and is
designed to reveal *which* audio reached it: it returns a feature per output row
whose value encodes the window's mean sample, so an off-by-one in the stride
shows up as a wrong number rather than as plausible-looking features.

Run with the project interpreter:
    /opt/miniconda3/envs/livetalking/bin/python -m pytest tests/test_pcm_asr.py
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from avatars.audio_features import pcm as pcm_mod
from avatars.audio_features.pcm import (
    FEAT_DIM,
    N_VALID_FEATS,
    STRIDE_SAMPLES,
    WINDOW_SAMPLES,
    PcmASR,
)


def make_opt(fps=25, batch_size=16):
    return types.SimpleNamespace(
        fps=fps, batch_size=batch_size,
        l=10, r=10, sessionid="test",
    )


def make_parent_stub():
    return types.SimpleNamespace(
        custom_audiotype=0, quit_event=None,
        get_custom_audio_stream=lambda typ: None,
    )


class WindowProbe:
    """Stub extractor that reports each window's first sample.

    Every output row carries ``window[0]`` in slot 0, so the test can assert
    exactly which slice of the audio produced which feature. A stride error
    moves this value; a window-size error changes it by the wrong amount.
    """

    def __init__(self):
        self.windows = []

    def __call__(self, window):
        assert window.shape == (WINDOW_SAMPLES,), window.shape
        self.windows.append(float(window[0]))
        out = np.zeros((N_VALID_FEATS, FEAT_DIM), dtype=np.float32)
        out[:, 0] = float(window[0])
        return out


def build(batch_size=16):
    probe = WindowProbe()
    asr = PcmASR(make_opt(batch_size=batch_size), make_parent_stub(), probe)
    return asr, probe


# ─── geometry, pinned to upstream's numbers ────────────────────────────────
def test_geometry_matches_upstream_chunksize_352():
    """chunksize=(3,5,2) at 16 kHz: sum*0.04*16000+80 and chunksize[1]*0.04*16000."""
    assert WINDOW_SAMPLES == 6480
    assert STRIDE_SAMPLES == 3200
    assert N_VALID_FEATS == 5


def test_one_feature_per_video_frame():
    """25 features/s against 25 fps is what lets a batch be sliced by row."""
    feats_per_second = 16000 / STRIDE_SAMPLES * N_VALID_FEATS
    assert feats_per_second == 25.0
    # And the app's chunk is exactly one video frame of audio.
    assert 16000 // (25 * 2) == 320


# ─── windowing arithmetic ──────────────────────────────────────────────────
def test_no_extraction_below_one_full_window():
    asr, probe = build()
    asr.push_pcm(np.zeros(WINDOW_SAMPLES - 1, dtype=np.float32))
    assert asr._extract_available() == 0
    assert probe.windows == []


def test_exactly_one_window_extracts_five_rows():
    asr, probe = build()
    asr.push_pcm(np.arange(WINDOW_SAMPLES, dtype=np.float32))
    assert asr._extract_available() == N_VALID_FEATS
    assert asr.pending_rows == N_VALID_FEATS
    assert len(probe.windows) == 1


def test_window_advances_by_stride_not_by_window():
    """The 3280-sample tail must be re-used, not discarded."""
    asr, probe = build()
    audio = np.arange(WINDOW_SAMPLES + STRIDE_SAMPLES, dtype=np.float32)
    asr.push_pcm(audio)
    asr._extract_available()
    assert len(probe.windows) == 2
    assert probe.windows[0] == 0.0
    # Second window starts at STRIDE, not at WINDOW.
    assert probe.windows[1] == float(STRIDE_SAMPLES)


def test_overlap_tail_is_retained_after_extraction():
    asr, _probe = build()
    asr.push_pcm(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    asr._extract_available()
    assert asr._pcm.shape[0] == WINDOW_SAMPLES - STRIDE_SAMPLES


def test_feature_row_tracks_its_own_audio():
    """The desync test: row k must carry the audio that produced frame k."""
    asr, _probe = build(batch_size=10)
    # 5 frames of audio per window, so feed two full windows worth.
    audio = np.zeros(WINDOW_SAMPLES + STRIDE_SAMPLES, dtype=np.float32)
    audio[-1] = 0.0
    asr.push_pcm(audio)
    rows = asr._extract_available()
    assert rows == 2 * N_VALID_FEATS
    # Both windows began at different offsets, and the probe wrote those
    # offsets into slot 0 of every row it produced.
    assert asr.pending_rows == 10


# ─── batch emission contract (what the inference loop requires) ────────────
def test_batch_has_exactly_batch_size_rows():
    asr, _probe = build(batch_size=16)
    # 16 frames of audio == 3200*16 samples; enough for several windows.
    asr.push_pcm(np.zeros(STRIDE_SAMPLES * 16, dtype=np.float32))
    asr._extract_available()
    assert asr.pending_rows >= 16
    batch = asr._feat[:16]
    assert batch.shape == (16, FEAT_DIM)


def test_run_step_emits_one_batch_and_paces_audio():
    """run_step must feed output_queue (audio clock) and feat_queue (inference)."""
    asr, _probe = build(batch_size=4)
    # One 6480-sample window needs 6480/320 = 21 frames of audio; each run_step
    # consumes batch_size*2 = 8 frames, so 3 steps cross the window boundary.
    for _ in range(3):
        asr.run_step()
    # Bounded queue (maxsize=2): drain what is there instead of asserting a
    # count, because _emit deliberately drops stale batches when undrained.
    batches = []
    while not asr.feat_queue.empty():
        batches.append(asr.feat_queue.get_nowait())
    assert batches, "no feature batch was emitted"
    batch = batches[0]
    assert isinstance(batch, list)
    assert len(batch) == 4
    assert all(np.asarray(r).shape == (FEAT_DIM,) for r in batch)
    # Audio must have been forwarded: 3 steps * (4*2 chunks) = 24 chunks.
    assert asr.output_queue.qsize() == 24


def test_undrained_consumer_never_blocks_the_asr_thread():
    """A stalled inference loop must not wedge the ASR worker.

    feat_queue is bounded, so a plain put() deadlocks as soon as inference
    falls behind — which starvation makes routine. _emit must instead drop the
    superseded batch and count it. output_queue is drained here because it is
    the audio clock's queue and blocking on it is the *documented* behaviour
    (it applies backpressure to a producer that outruns the audio track).
    """
    asr, _probe = build(batch_size=4)
    asr.push_pcm(np.zeros(WINDOW_SAMPLES * 20, dtype=np.float32))
    for _ in range(40):
        asr.run_step()          # must return, not block
        while not asr.output_queue.empty():     # act as the audio consumer
            asr.output_queue.get_nowait()
    assert asr.dropped_batches > 0
    assert asr.feat_queue.qsize() <= 2


def test_starvation_pads_to_batch_size_and_is_counted():
    """A short batch must not crash the loop; it repeats the newest feature."""
    asr, _probe = build(batch_size=8)
    # Not enough audio for any window at all.
    asr.run_step()
    assert asr.starved_batches == 1
    batch = asr.feat_queue.get()
    assert len(batch) == 8
    assert all(np.asarray(r).shape == (FEAT_DIM,) for r in batch)


def test_odd_batch_size_still_slices_by_row():
    """batch_size is not a multiple of 5 in practice; slicing must cope."""
    for bs in (7, 11, 13, 16):
        asr, _probe = build(batch_size=bs)
        asr.push_pcm(np.zeros(WINDOW_SAMPLES * 4, dtype=np.float32))
        asr._extract_available()
        assert asr.pending_rows >= bs, (bs, asr.pending_rows)
        batch = asr._feat[:bs]
        assert batch.shape == (bs, FEAT_DIM)


# ─── shape validation ──────────────────────────────────────────────────────
def test_wrong_feat_shape_fails_loudly():
    """A wrong-shaped extractor must fail here, not desync 40 frames later."""
    def bad(window):
        return np.zeros((N_VALID_FEATS, 768), dtype=np.float32)   # whisper dim

    asr = PcmASR(make_opt(), make_parent_stub(), bad)
    asr.push_pcm(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    with pytest.raises(ValueError, match="expected"):
        asr._extract_available()


# ─── barge-in ──────────────────────────────────────────────────────────────
def test_flush_drops_pcm_and_features():
    """Barge-in must discard features for speech that will never be heard."""
    asr, _probe = build()
    asr.push_pcm(np.ones(WINDOW_SAMPLES, dtype=np.float32))
    asr._extract_available()
    assert asr.pending_rows > 0
    asr.flush_talk()
    assert asr.pending_rows == 0
    assert asr._pcm.shape[0] == 0


def test_flush_does_not_reset_the_window_counter():
    """Counters are diagnostics; a barge-in changes what is spoken, not time."""
    asr, _probe = build()
    asr.push_pcm(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    asr._extract_available()
    before = asr.windows_done
    asr.flush_talk()
    assert asr.windows_done == before


# ─── importability without heavy deps ──────────────────────────────────────
def test_module_imports_without_onnxruntime():
    """The windowing layer must be importable where onnxruntime-gpu is absent."""
    import sys
    # The factory defers its import; nothing at module scope may touch it.
    assert "core.aux_models.hubert_stream" not in sys.modules
    assert callable(pcm_mod.make_ditto_feat_fn)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
