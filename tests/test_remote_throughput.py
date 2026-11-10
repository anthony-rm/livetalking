"""Tests for the three remote-throughput changes.

1. compact audio-feature block   (upload 3.3x smaller, must be lossless)
2. JPEG quality default 75
3. pipelined inference           (order-preserving, one batch overlapped)

The block form is the dangerous one: a wrong offset would not crash, it would
attach the wrong audio window to frames and produce subtly wrong lips. So the
central test does not trust the arithmetic -- it takes the REAL whisper chunks
the ASR stage produces and asserts the block reconstructs them byte for byte.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest


# ─── 1. compact feature block ─────────────────────────────────────────────

def _real_chunks(batch_size=8, l=10, r=10):
    """Chunks exactly as WhisperASR.run_step() builds them, from real whisper.

    Uses the actual audio processor so a change in either the window arithmetic
    or the model's output shape fails here rather than on the GPU.
    """
    from avatars.musetalk.whisper.audio2feature import Audio2Feature
    from avatars.audio_features.whisper import WhisperASR
    import avatars.audio_features.base_asr as base

    ap = Audio2Feature(model_path="./models/whisper")
    rng = np.random.default_rng(0)
    n_chunks = batch_size * 2 + l + r
    audio = (rng.standard_normal((n_chunks, 320)) * 0.05).astype(np.float32)
    feature = ap.audio2feat(audio)

    stub = base.BaseASR.__new__(base.BaseASR)
    stub.fps, stub.batch_size = 25, batch_size
    stub.stride_left_size, stub.stride_right_size = l, r
    for name in ("_feature2chunks", "_chunk_spans", "feature_block",
                 "_attach_block"):
        setattr(stub, name, getattr(WhisperASR, name).__get__(stub))
    chunks = stub._feature2chunks(feature_array=feature, batch_size=batch_size,
                                  audio_feat_win=[2, 3], start=l / 2,
                                  feature_idx_multiplier=2)
    return stub, feature, chunks


def test_block_reconstructs_real_chunks_byte_identically():
    stub, feature, chunks = _real_chunks()
    batch = stub._attach_block(feature, chunks, [2, 3], 5, 2)

    assert batch.block is not None, f"refused: {batch.why_not}"
    rebuilt = [batch.block[o:o + batch.rows_per_frame].reshape(c.shape)
               for o, c in zip(batch.offsets, chunks)]
    for i, (got, want) in enumerate(zip(rebuilt, chunks)):
        assert np.array_equal(got, want), f"frame {i} differs"


def test_block_is_smaller_by_the_expected_factor():
    stub, feature, chunks = _real_chunks(batch_size=8)
    batch = stub._attach_block(feature, chunks, [2, 3], 5, 2)

    legacy = sum(c.nbytes for c in chunks)
    saved = legacy / batch.block.nbytes
    # 10-row windows advancing 2 rows: 8 frames need 24 rows, not 80.
    assert 3.0 < saved < 3.6, saved


def test_block_is_a_list_so_existing_consumers_keep_working():
    """Every consumer indexes/iterates/np.stack()s the batch."""
    stub, feature, chunks = _real_chunks(batch_size=4)
    batch = stub._attach_block(feature, chunks, [2, 3], 5, 2)

    assert isinstance(batch, list)
    assert len(batch) == 4
    assert np.stack(batch).shape[0] == 4
    assert batch[0] is chunks[0]


def test_block_refused_when_windows_are_ragged():
    """A backend with non-uniform windows must not get a bogus block."""
    stub, feature, _ = _real_chunks(batch_size=2)
    chunks = [np.zeros((10, 5, 384), np.float32),
              np.zeros((12, 5, 384), np.float32)]
    spans = [(0, 10), (10, 22)]        # 10 then 12 rows
    block, offsets, rows, why = stub.feature_block(feature, chunks, spans, 10)
    assert block is None
    assert "ragged" in why


def test_block_refused_at_the_array_edge():
    """Clamped windows repeat a row, so they are not a contiguous reshape."""
    stub, feature, _ = _real_chunks(batch_size=2)
    chunks = [np.zeros((10, 5, 384), np.float32)] * 2
    spans = [(-1, 9), (1, 11)]
    block, _o, _r, why = stub.feature_block(feature, chunks, spans, 10)
    assert block is None
    assert "edge" in why


def test_block_refused_when_values_differ():
    """The proof step: a block that reshapes but does not MATCH is rejected."""
    stub, feature, chunks = _real_chunks(batch_size=2)
    spans = [(0, 10), (2, 12)]
    # Same shapes as the real chunks, but one chunk's values are corrupted, so
    # the reshape succeeds and only the equality check can catch it.
    corrupted = [chunks[0].copy(), chunks[1].copy() + 0.5]
    block, _o, _r, why = stub.feature_block(feature, corrupted, spans, 10)
    assert block is None
    assert "differ" in why, why


# ─── worker side: rebuild ─────────────────────────────────────────────────

def _servicer():
    from remote.worker import RemoteWorkerServicer
    from remote.proto import inference_pb2_grpc  # noqa: F401  (import side effect)
    obj = RemoteWorkerServicer.__new__(RemoteWorkerServicer)
    obj.backend = None
    obj.backend_name = ""
    obj.avatar_id = ""
    return obj


def test_worker_rebuilds_block_into_per_frame_windows():
    from remote.proto import inference_pb2
    from remote.tensors import to_tensor

    # The worker's rule: frame = window.reshape(-1, 384), i.e. every leading dim
    # is flattened into the row axis and the columns stay 384 wide (the musetalk
    # UNet's d_model). A (rows, 768) block must become (2*rows, 384), NOT
    # (rows, 768) -- getting this wrong is what the 1920-vs-384 error was.
    rows = 10
    block = np.arange(rows * 768, dtype=np.float32).reshape(rows, 2, 384)
    request = inference_pb2.InferRequest(
        index=0, batch_size=1,
        feat_block=to_tensor(block), feat_offsets=[0], feat_rows_per_frame=rows)

    feats = _servicer()._request_features(request)
    assert len(feats) == 1
    assert feats[0].shape == (rows * 2, 384), feats[0].shape
    assert np.array_equal(feats[0], block[0:rows].reshape(-1, 384))


def test_worker_rejects_a_window_that_is_not_a_multiple_of_the_feature_width():
    from remote.proto import inference_pb2
    from remote.tensors import to_tensor

    request = inference_pb2.InferRequest(
        index=0, batch_size=1,
        feat_block=to_tensor(np.zeros((10, 7), np.float32)),
        feat_offsets=[0], feat_rows_per_frame=10)
    with pytest.raises(ValueError, match="not a multiple"):
        _servicer()._request_features(request)


def test_worker_still_accepts_legacy_per_frame_tensors():
    """An un-updated host must keep working: no block, plain tensors."""
    from remote.proto import inference_pb2
    from remote.tensors import to_tensor

    a = np.ones((10, 3, 4), np.float32)
    request = inference_pb2.InferRequest(
        index=0, batch_size=2, audiofeat=[to_tensor(a), to_tensor(a * 2)])
    feats = _servicer()._request_features(request)
    assert np.array_equal(feats[0], a)
    assert np.array_equal(feats[1], a * 2)


def test_worker_rejects_a_block_whose_windows_exceed_it():
    from remote.proto import inference_pb2
    from remote.tensors import to_tensor

    request = inference_pb2.InferRequest(
        index=0, batch_size=1, feat_block=to_tensor(np.zeros((4, 3, 4), np.float32)),
        feat_offsets=[2], feat_rows_per_frame=10)
    with pytest.raises(ValueError, match="outside"):
        _servicer()._request_features(request)


def test_worker_rejects_a_block_with_the_wrong_frame_count():
    from remote.proto import inference_pb2
    from remote.tensors import to_tensor

    request = inference_pb2.InferRequest(
        index=0, batch_size=3,
        feat_block=to_tensor(np.zeros((30, 384), np.float32)),
        feat_offsets=[0, 2], feat_rows_per_frame=10)
    with pytest.raises(ValueError, match="yielded 2"):
        _servicer()._request_features(request)


# ─── capability negotiation ───────────────────────────────────────────────

def _client_with(capabilities):
    from remote.client import RemoteClient
    c = RemoteClient.__new__(RemoteClient)
    c.address = "test:1"
    c.capabilities = frozenset(capabilities)
    return c


def test_client_sends_the_block_only_when_the_worker_advertises_it():
    from avatars.audio_features.whisper import FeatBatch

    batch = FeatBatch([np.zeros((10, 384), np.float32)])
    batch.block = np.zeros((10, 384), np.float32)
    batch.offsets = (0,)
    batch.rows_per_frame = 10

    modern = _client_with(["feat-block"])._build_request(
        0, 1, batch, encoding="jpeg-bgr", quality=75, feature_dtype="float16")
    assert modern.feat_block.data
    assert list(modern.feat_offsets) == [0]

    legacy = _client_with([])._build_request(
        0, 1, batch, encoding="jpeg-bgr", quality=75, feature_dtype="float16")
    assert not legacy.feat_block.data
    assert len(legacy.audiofeat) == 1


def test_client_sends_legacy_tensors_when_the_batch_has_no_block():
    """The refusal path in WhisperASR must still produce a valid request."""
    from avatars.audio_features.whisper import FeatBatch

    batch = FeatBatch([np.zeros((10, 384), np.float32)] * 2)
    req = _client_with(["feat-block"])._build_request(
        0, 2, batch, encoding="jpeg-bgr", quality=75, feature_dtype="float16")
    assert not req.feat_block.data
    assert len(req.audiofeat) == 2


def test_default_jpeg_quality_is_75():
    import inspect
    from remote.client import RemoteClient
    sig = inspect.signature(RemoteClient.infer_batch)
    assert sig.parameters["quality"].default == 75


def test_worker_advertises_its_capabilities():
    from remote.worker import RemoteWorkerServicer
    assert "feat-block" in RemoteWorkerServicer.CAPABILITIES
    assert "jpeg-bgr" in RemoteWorkerServicer.CAPABILITIES


# ─── 3. pipelining ────────────────────────────────────────────────────────

class FakeClient:
    """Stands in for RemoteClient with a controllable latency per batch."""

    def __init__(self, delays, fail_on=()):
        self.delays = delays
        self.fail_on = set(fail_on)
        self.calls = []
        self.timeout = 30.0
        self._lock = threading.Lock()
        self.stub = self                     # PipelinedInfer calls .stub
        self.concurrent_peak = 0
        self._concurrent = 0

    # Worker entry point is _stub.InferBatch; PipelinedInfer calls it directly.
    @property
    def _stub(self):
        return self

    def _build_request(self, index, batch_size, batch, **kw):
        return {"index": index}

    def InferBatch(self, request, timeout=None):  # noqa: N802
        index = request["index"]
        with self._lock:
            self.calls.append(index)
            self._concurrent += 1
            self.concurrent_peak = max(self.concurrent_peak, self._concurrent)
        try:
            time.sleep(self.delays(index))
            if index in self.fail_on:
                raise RuntimeError(f"batch {index} exploded")
            return _FakeReply(index)
        finally:
            with self._lock:
                self._concurrent -= 1


class _FakeReply:
    def __init__(self, index):
        self.encoding = ""
        self.frames = None
        self._index = index


@pytest.fixture(autouse=True)
def _patch_decode(monkeypatch):
    """_decode_reply would need real frames; return the batch index instead."""
    import remote.client as client_mod
    monkeypatch.setattr(client_mod, "_decode_reply",
                        lambda reply: np.array([reply._index]))


def test_pipeline_overlaps_two_batches():
    """Two RPCs must genuinely be in flight at once, or nothing is gained."""
    from remote.client import PipelinedInfer

    client = FakeClient(delays=lambda i: 0.15)
    pipe = PipelinedInfer(client, depth=2)
    try:
        # Submit two before collecting either: that is the overlap.
        s0 = pipe.submit(0, 1, [np.zeros((1, 1))])
        s1 = pipe.submit(1, 1, [np.zeros((1, 1))])
        r0 = pipe.result(s0)
        r1 = pipe.result(s1)
    finally:
        pipe.close()

    assert r0[0] == 0 and r1[0] == 1
    assert client.concurrent_peak >= 2, "no actual overlap happened"


def test_pipeline_returns_frames_in_submission_order_even_if_they_finish_out_of_order():
    from remote.client import PipelinedInfer

    # Batch 0 is SLOW, batch 1 is fast: results arrive reversed.
    client = FakeClient(delays=lambda i: 0.30 if i == 0 else 0.01)
    pipe = PipelinedInfer(client, depth=2)
    try:
        s0 = pipe.submit(0, 1, [np.zeros((1, 1))])
        s1 = pipe.submit(1, 1, [np.zeros((1, 1))])
        assert pipe.result(s0)[0] == 0        # must not yield 1 first
        assert pipe.result(s1)[0] == 1
    finally:
        pipe.close()


def test_pipeline_failure_surfaces_and_does_not_shift_later_batches():
    from remote.client import PipelinedInfer, RemoteWorkerError

    client = FakeClient(delays=lambda i: 0.02, fail_on={0})
    pipe = PipelinedInfer(client, depth=2)
    try:
        s0 = pipe.submit(0, 1, [np.zeros((1, 1))])
        s1 = pipe.submit(1, 1, [np.zeros((1, 1))])
        with pytest.raises(RemoteWorkerError, match="exploded"):
            pipe.result(s0)
        # Batch 1 is still addressable: a failed sibling must not consume it.
        assert pipe.result(s1)[0] == 1
    finally:
        pipe.close()


def test_pipeline_close_unblocks_a_waiter():
    from remote.client import PipelinedInfer, RemoteWorkerError

    client = FakeClient(delays=lambda i: 5.0)
    pipe = PipelinedInfer(client, depth=1)
    result = {}

    def wait():
        try:
            pipe.result(pipe.submit(0, 1, [np.zeros((1, 1))]))
        except RemoteWorkerError as exc:
            result["error"] = str(exc)

    t = threading.Thread(target=wait)
    t.start()
    time.sleep(0.2)
    pipe.close()
    t.join(timeout=5)
    assert not t.is_alive(), "close() did not wake the waiter"
    assert "closed" in result.get("error", "")


def test_pipeline_depth_one_is_sequential():
    from remote.client import PipelinedInfer

    client = FakeClient(delays=lambda i: 0.05)
    pipe = PipelinedInfer(client, depth=1)
    try:
        for i in range(3):
            assert pipe.run(i, 1, [np.zeros((1, 1))])[0] == i
    finally:
        pipe.close()
    assert client.concurrent_peak == 1


# ─── base_avatar wiring ───────────────────────────────────────────────────

def test_base_avatar_uses_the_synchronous_path_without_a_pipeline():
    """Local backends must be untouched by the pipelining change."""
    from avatars.base_avatar import BaseAvatar

    stub = BaseAvatar.__new__(BaseAvatar)
    assert stub._pipeline() is None
    sentinel = np.zeros((2, 2))
    stub.inference_batch = lambda index, batch: sentinel
    assert stub._resolve_batch(stub._submit_batch(0, sentinel)) is sentinel


def test_base_avatar_routes_batches_through_the_pipeline_when_present():
    from avatars.base_avatar import BaseAvatar
    from remote.client import PipelinedInfer

    stub = BaseAvatar.__new__(BaseAvatar)
    stub.batch_size = 1
    stub.infer_pipeline = PipelinedInfer(FakeClient(delays=lambda i: 0.01), depth=1)
    try:
        assert stub._resolve_batch(stub._submit_batch(7, [np.zeros((1, 1))]))[0] == 7
    finally:
        stub.infer_pipeline.close()


def test_inference_loop_emits_every_batch_in_order():
    """Drive the real loop end to end: N batches in, N*batch_size frames out."""
    import queue as _queue
    from avatars.base_avatar import BaseAvatar, AudioFrameData
    from utils.image import mirror_index

    batch_size = 2
    stub = BaseAvatar.__new__(BaseAvatar)
    stub.batch_size = batch_size
    stub.custom_index = {}
    stub.quit_event = None
    stub.res_frame_queue = _queue.Queue(1000)
    stub.frame_list_cycle = [np.zeros((2, 2, 3), np.uint8)] * 4
    stub.infer_pipeline = None
    stub._resolved = []

    class ASR:
        chunk = 320
        def __init__(self):
            self.feat_queue = _queue.Queue()
            self.output_queue = _queue.Queue()
            for _ in range(3):
                self.feat_queue.put([np.zeros((10, 384), np.float32)] * batch_size)
                for _ in range(batch_size * 2):
                    self.output_queue.put(
                        AudioFrameData(data=np.zeros(320, np.float32), type=0))

    stub.asr = ASR()

    def inference_batch(index, batch):
        stub._resolved.append(index)
        return [np.full((1, 1, 3), index, np.uint8)] * batch_size

    stub.inference_batch = inference_batch
    stub.get_avatar_length = lambda: 4

    quit_event = threading.Event()
    t = threading.Thread(target=stub.inference, args=(quit_event,), daemon=True)
    t.start()
    deadline = time.time() + 10
    while stub.res_frame_queue.qsize() < 3 * batch_size and time.time() < deadline:
        time.sleep(0.01)
    quit_event.set()
    t.join(timeout=5)
    assert not t.is_alive(), "inference loop did not stop"

    assert stub._resolved == [0, 2, 4], stub._resolved
    idxs = []
    while not stub.res_frame_queue.empty():
        _frame, _audio, idx = stub.res_frame_queue.get()
        idxs.append(idx)
    # mirror_index is a ping-pong bounce, not a wraparound: past the end it
    # walks back. Expected values are computed, not guessed.
    assert idxs == [mirror_index(4, i) for i in range(6)], idxs
    assert len(idxs) == 3 * batch_size
