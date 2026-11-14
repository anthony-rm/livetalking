###############################################################################
#  Ditto audio features — raw PCM in, HuBERT features out
#
#  Ditto's LMDM was trained on features from *its own* HuBERT checkpoint
#  (`hubert_streaming_fix_kv.onnx`), not on this repo's `HubertASR`
#  (TencentGameMate/chinese-hubert) and not on whisper. Feeding a different
#  feature space into the motion model produces plausible-looking but wrong
#  lips, so this module deliberately forwards PCM and lets Ditto's own
#  extractor run. See PLAN_DITTO.md §8.2.
###############################################################################
"""Streaming PCM → Ditto HuBERT features.

Geometry (all verified against upstream `core/atomic_components/wav2feat.py`
and `stream_pipeline_online.py`, not assumed):

    window   6480 samples  = 0.405 s   (``sum(chunksize)*0.04*16000 + 80``)
    stride   3200 samples  = 0.200 s   (``chunksize[1] * 0.04 * 16000``)
    output   5 features    = [5, 1024] (``reshape(chunksize[1], 2, 1024).mean(1)``)

5 features per 0.200 s is 25 features/second, and video is 25 fps, so this is
**exactly one feature per video frame** — which is why the app's per-frame batch
contract maps onto it with no resampling.

Two things this class deliberately does NOT do:

* It does not seed Ditto's left context. Upstream seeds ``overlap_v2`` (70)
  zero features before the first real one; the avatar backend owns that, because
  only it knows ``overlap_v2`` from the cfg. This class reports honest frame
  indices starting at 0.
* It does not window for the LMDM. Upstream's 80-frame sliding window is driven
  inside ``Audio2Motion``; duplicating it here would give two sources of truth
  for the same counter.
"""
from __future__ import annotations

import queue
from queue import Queue
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from avatars.audio_features.base_asr import BaseASR
from utils.logger import logger


# Geometry, as module constants so tests and the backend can assert on them
# rather than re-deriving the numbers from source.
WINDOW_SAMPLES = 6480
STRIDE_SAMPLES = 3200
N_VALID_FEATS = 5
FEAT_DIM = 1024

#: ``feat_fn(window) -> (N_VALID_FEATS, FEAT_DIM)``, called with a float32
#: window of exactly ``WINDOW_SAMPLES`` samples. Injected so the windowing
#: arithmetic can be tested without the 1.46 GB HuBERT checkpoint.
FeatFn = Callable[[NDArray[np.float32]], NDArray[np.float32]]


class PcmASR(BaseASR):
    """Turns the app's 20 ms PCM chunks into one feature per video frame.

    The app calls `put_audio_frame` with 320-sample chunks and drains
    `feat_queue`, one item per inference batch, each a list of `batch_size`
    per-frame features. Between those two lies a mismatch:

    * this extractor emits features in groups of 5 (one per 0.2 s of audio),
    * a batch wants an arbitrary `batch_size` rows.

    `batch_size` is usually not a multiple of 5, so features are buffered and
    sliced by row instead of being forwarded group-by-group. Slicing (rather
    than re-windowing per frame) is what keeps frame *n* paired with the exact
    audio that produced it.
    """

    WINDOW = WINDOW_SAMPLES
    STRIDE = STRIDE_SAMPLES
    N_VALID = N_VALID_FEATS
    FEAT_DIM = FEAT_DIM

    def __init__(self, opt, parent, feat_fn: FeatFn):
        super().__init__(opt, parent)
        self.feat_fn = feat_fn

        # Un-extracted PCM tail. Grows by `chunk` per frame, shrinks by `STRIDE`
        # per extraction; it stays >= STRIDE after the first window.
        self._pcm: NDArray[np.float32] = np.zeros((0,), dtype=np.float32)

        # Extracted-but-not-yet-emitted feature rows, oldest first.
        self._feat: NDArray[np.float32] = np.zeros((0, self.FEAT_DIM), dtype=np.float32)

        self._closed = False
        self._windows_done = 0
        self._frames_emitted = 0
        # Starvation counter: a batch was short and padded. Surfaced so the
        # backend can log it instead of shipping a silent audio/video desync.
        self.starved_batches = 0
        # Batches discarded because the inference loop did not drain fast
        # enough. Non-zero means the render path is the bottleneck.
        self.dropped_batches = 0

    # ─── introspection (used by tests and the backend's health log) ────────
    @property
    def windows_done(self) -> int:
        """How many HuBERT windows have been extracted (== frames/5, roughly)."""
        return self._windows_done

    @property
    def frames_emitted(self) -> int:
        return self._frames_emitted

    @property
    def pending_rows(self) -> int:
        """Feature rows extracted but not yet handed to a batch."""
        return int(self._feat.shape[0])

    # ─── PCM in ────────────────────────────────────────────────────────────
    def push_pcm(self, pcm: NDArray[np.float32]) -> None:
        """Append raw 16 kHz mono PCM. Split out so tests skip the queue."""
        if pcm.size:
            self._pcm = np.concatenate(
                [self._pcm, np.asarray(pcm, dtype=np.float32).reshape(-1)])

    def _extract_available(self) -> int:
        """Run the extractor over every complete window; return rows produced."""
        produced = 0
        while self._pcm.shape[0] >= self.WINDOW:
            window = self._pcm[:self.WINDOW]
            feats = np.asarray(self.feat_fn(window), dtype=np.float32)
            # A wrong-shaped return would surface 40 frames later as a silent
            # desync, so fail at the boundary instead.
            if feats.shape != (self.N_VALID, self.FEAT_DIM):
                raise ValueError(
                    f"feat_fn returned {feats.shape}, expected "
                    f"({self.N_VALID}, {self.FEAT_DIM}) — check chunksize")
            self._feat = np.concatenate([self._feat, feats], axis=0)
            # Advance by STRIDE, not WINDOW: the last WINDOW-STRIDE samples are
            # the overlap carried into the next window.
            self._pcm = self._pcm[self.STRIDE:]
            self._windows_done += 1
            produced += self.N_VALID
        return produced

    # ─── one iteration of the ASR worker ───────────────────────────────────
    def run_step(self) -> None:
        """Consume audio, extract what is possible, emit at most one batch.

        Mirrors `WhisperASR.run_step`: `batch_size*2` 20 ms chunks in, one
        `batch_size`-row item out, and the same chunks into `output_queue` so
        the audio track keeps its own 50 Hz clock.
        """
        for _ in range(self.batch_size * 2):
            audio_frame = self.get_audio_frame()
            self.frames.append(audio_frame.data)
            self.output_queue.put(audio_frame)
            self.push_pcm(audio_frame.data)

        self._extract_available()

        n = self.batch_size
        if self._feat.shape[0] >= n:
            batch = self._feat[:n]
            self._feat = self._feat[n:]
            self._frames_emitted += n
            self._emit([row.copy() for row in batch])
        else:
            # Underfed: pad with the newest row so the batch still has the
            # shape the inference loop requires, and count it. A starved batch
            # repeats the last feature (a held mouth) rather than emitting
            # garbage; the backend's health log reports the count.
            self.starved_batches += 1
            if self._feat.shape[0]:
                last = self._feat[-1]
                pad = np.repeat(last[None, :], n - self._feat.shape[0], axis=0)
                batch = np.concatenate([self._feat, pad], axis=0)
                self._feat = np.zeros((0, self.FEAT_DIM), dtype=np.float32)
            else:
                batch = np.zeros((n, self.FEAT_DIM), dtype=np.float32)
            self._frames_emitted += n
            self._emit([row for row in batch])

    def _emit(self, batch: list) -> None:
        """Queue one batch WITHOUT blocking the ASR worker.

        ``feat_queue`` is bounded (maxsize=2) and drained by the inference
        loop. A plain ``put`` here deadlocks the moment inference falls behind
        ASR — which starvation makes routine, because a starved batch is emitted
        even when no audio arrived. The consumer holds at most one batch in
        flight, so an undrainable queue means the oldest batch is already
        superseded: drop it and count the loss, the way audio frames are
        dropped rather than queued forever elsewhere in this pipeline.
        """
        try:
            self.feat_queue.put_nowait(batch)
        except queue.Full:
            self.dropped_batches += 1
            try:
                self.feat_queue.get_nowait()        # discard the stale batch
            except queue.Empty:
                pass
            try:
                self.feat_queue.put_nowait(batch)
            except queue.Full:
                # Consumer is mid-swap; the newer batch wins next round.
                self.dropped_batches += 1

    # ─── shutdown ──────────────────────────────────────────────────────────
    def flush_talk(self) -> None:
        """Drop queued audio AND un-extracted PCM.

        Barge-in means the audio the user is about to hear is different, so
        features derived from the old PCM are not merely stale — they belong to
        a sentence that will never be spoken. Keeping them would lip-sync to
        silence.
        """
        super().flush_talk()
        self._pcm = np.zeros((0,), dtype=np.float32)
        self._feat = np.zeros((0, self.FEAT_DIM), dtype=np.float32)


def make_ditto_feat_fn(
    model_path: str,
    device: str = "cuda",
) -> FeatFn:
    """Wrap Ditto's own HuBERT ONNX as a plain ``window -> features`` callable.

    Kept out of `PcmASR` so the windowing logic is importable (and testable) on
    a machine without onnxruntime-gpu or the 1.46 GB checkpoint. Upstream's
    `HubertStreaming` owns a streaming KV cache, so one instance is stateful and
    must not be shared between sessions.
    """
    from core.aux_models.hubert_stream import HubertStreaming

    hubert = HubertStreaming(model_path=model_path, device=device)

    def feat_fn(window: NDArray[np.float32]) -> NDArray[np.float32]:
        # Verbatim geometry from Wav2FeatHubert.__call__: the KV-cache run
        # returns 8 raw frames for chunksize (3,5,2); frames 1..5 are valid and
        # each pair is averaged, which is what `reshape(5,2,1024).mean(1)` does.
        chunksize = (3, 5, 2)
        valid_feat_s = -sum(chunksize[1:]) * 2      # -7
        valid_feat_e = -chunksize[2] * 2            # -2
        encoding = hubert(window)
        valid = encoding[valid_feat_s:valid_feat_e]
        return valid.reshape(chunksize[1], 2, FEAT_DIM).mean(1)

    return feat_fn
