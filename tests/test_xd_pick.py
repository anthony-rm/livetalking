"""_pick_xd: one motion row per rendered frame, anchored once per pointer move.

The content-locked picker exists so drop-mode speech does not show motion for
audio that already played. Its coupling to the batch loop is what went wrong:
the picker is called `batch_size` times per batch while `_paced_chunk_idx` does
not move, so re-resolving the same target on every call consumed one extra row
per batch. The queue then starved and the mouth fell back to the neutral pose —
the registration face, mid-sentence (measured: 141 of 160 speaking frames).

These tests pin the contract without a GPU or model weights.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Asr:
    audio_lead_chunks = 10


def _owner(rows, paced):
    """A minimal DittoReal with only what _pick_xd touches."""
    from avatars.ditto_avatar import DittoReal

    owner = DittoReal.__new__(DittoReal)
    owner.asr = _Asr()
    owner._paced_chunk_idx = paced
    owner._idle_xd = {"neutral": True}
    owner._x_d_pending = [{"_row": r, "k": r} for r in rows]
    owner._xd_trace_n = 10_000       # silence the trace branch
    owner._neutral_picks = 0
    return owner


def test_one_row_per_frame_within_a_batch():
    """10 picks from one stationary pointer must consume exactly 10 rows.

    The old picker re-resolved the same target on every call: the first pick
    dropped the stale prefix (best+1 rows for ONE frame), so a batch ate 11
    rows while the LMDM emits 10. The queue starved by one row per batch and
    that frame showed the neutral pose.
    """
    rows = list(range(100, 130))
    owner = _owner(rows, paced=240)
    # Steady state: the pointer was already anchored by the previous batch, so
    # this batch pays no stale-prefix drop. (The one-time prefix drop is
    # covered by test_pointer_advance_re_anchors_and_drops_stale_rows.)
    owner._xd_anchor_played = 240
    before = len(owner._x_d_pending)

    picked = [owner._pick_xd() for _ in range(10)]

    consumed = before - len(owner._x_d_pending)
    assert consumed == 10, (
        f"10 frames consumed {consumed} rows — the anchor is re-resolving "
        f"inside the batch")
    assert all(p.get("neutral") is None for p in picked), "neutral leaked in"
    got = [p["k"] for p in picked]
    assert len(set(got)) == 10, f"duplicate rows picked: {got}"
    assert got == sorted(got), f"rows out of order: {got}"
    assert owner._neutral_picks == 0


def test_no_neutral_fallback_when_queue_is_deep_enough():
    """A deep queue must never starve across a whole utterance.

    Reproduces the field case: 10 frames per batch, pointer advancing at the
    pacer's rate, LMDM emitting 10 rows per batch.
    """
    owner = _owner(range(0, 300), paced=0)
    neutrals = 0
    for batch in range(12):
        for _ in range(10):
            xd = owner._pick_xd()
            if xd.get("neutral"):
                neutrals += 1
        # The pacer advanced ~13 chunks while this batch rendered.
        owner._paced_chunk_idx += 13
        # And the LMDM emitted more rows.
        last = owner._x_d_pending[-1]["_row"] if owner._x_d_pending else 0
        owner._x_d_pending.extend(
            [{"_row": r, "k": r} for r in range(last + 1, last + 11)])

    assert neutrals == 0, f"{neutrals} neutral frames in a 120-frame utterance"


def test_pointer_advance_re_anchors_and_drops_stale_rows():
    """When the played pointer jumps, rows behind the new target go."""
    owner = _owner(range(0, 100), paced=0)
    owner._pick_xd()                       # anchors at the old pointer
    before = len(owner._x_d_pending)

    owner._paced_chunk_idx = 200           # audio jumped ahead (barge-in etc.)
    owner._pick_xd()

    remaining = [e["_row"] for e in owner._x_d_pending]
    assert len(owner._x_d_pending) < before, "stale rows were not dropped"
    assert all(r >= 95 for r in remaining), f"stale rows survived: {remaining}"


def test_empty_queue_falls_back_to_neutral_and_counts():
    owner = _owner([], paced=100)
    xd = owner._pick_xd()
    assert xd.get("neutral") is True
    assert owner._neutral_picks == 1
