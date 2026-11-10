"""Measure the remote pipeline's per-batch cost split: legacy vs compact wire.

Run it on the HOST, from anywhere, with the pinned interpreter:

    /opt/miniconda3/envs/livetalking/bin/python scripts/probe_remote_pipeline.py
    ... --batch-sizes 4,8,16 --formats legacy,block

What it answers
---------------
* ``ms/frame`` per wire format. A FLAT per-frame cost across batch sizes means
  the pipeline is bandwidth-bound; a large fixed per-batch term means per-RPC
  latency/encode overhead.
* The bytes ACTUALLY on the wire, measured from the built request and the reply
  (``len(blob)`` / ``.frames.data``) -- never from a docstring.
* The implied link rate. Compare with a raw ``dd | ssh 'cat > f'`` ceiling: if
  the RPC path gets well under it, the gap is stop-and-wait plus encode.

Two formats are compared because they are the two halves of this change:
  legacy = one tensor per frame (80 rows for a batch of 8)
  block  = one contiguous slice + per-frame offsets (24 rows), 3.33x smaller

Read-only with respect to model state: it calls Health and InferBatch on
whatever avatar the worker already has resident. It never calls
LoadModel/LoadAvatar/Unload, so it cannot evict a live session's weights.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Run as `python scripts/probe_remote_pipeline.py`, which puts scripts/ (not the
# repo root) on sys.path -- so `remote.*`/`utils.*` would not import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEAT_COLS = 384
ROW_WIDTH = 5 * 384          # one whisper output row, as the model emits it


def _address() -> str:
    from remote.settings import effective
    address, source = effective()
    if not address:
        print("No worker address configured (admin console / --remote_worker / "
              "LT_REMOTE_WORKER).", file=sys.stderr)
        raise SystemExit(2)
    print(f"worker: {address} (source: {source})")
    return address


def _make_batch(batch_size: int, rows: int, form: str):
    """A feature batch in the requested wire form, from REAL host machinery.

    ``legacy`` is what the old code sent: `batch_size` independent tensors of
    `rows` x 384 values. ``block`` is produced by the same
    ``WhisperASR._attach_block`` the live pipeline uses, so this measures the
    real block shape (span, hidden_states, 384) instead of a guess -- an
    earlier version of this script invented a 2-D block and the worker rebuilt
    it into the wrong shape.
    """
    from avatars.audio_features.whisper import FeatBatch, WhisperASR
    import avatars.audio_features.base_asr as base

    stride = 2                                    # musetalk advances 2 rows/frame
    span = rows + (batch_size - 1) * stride
    feature = np.random.default_rng(0).standard_normal(
        (span, 5, 384)).astype(np.float32)

    stub = base.BaseASR.__new__(base.BaseASR)
    stub.fps, stub.batch_size = 25, batch_size
    stub.stride_left_size, stub.stride_right_size = 10, 10
    for name in ("_feature2chunks", "_chunk_spans", "feature_block",
                 "_attach_block"):
        setattr(stub, name, getattr(WhisperASR, name).__get__(stub))
    chunks = stub._feature2chunks(feature_array=feature, batch_size=batch_size,
                                  audio_feat_win=[2, 3], start=5,
                                  feature_idx_multiplier=1)
    if form == "block":
        return stub._attach_block(feature, chunks, [2, 3], 5, 1)
    return FeatBatch(chunks)


def measure(client, batch_sizes, forms, reps, rows, encoding, quality):
    import grpc

    print(f"\nwire: encoding={encoding or 'raw'} quality={quality or '-'} "
          f"rows/frame={rows} capabilities={sorted(client.capabilities) or 'none'}")
    print(f"{'form':>7} {'bs':>3} {'best ms':>9} {'ms/frame':>9} "
          f"{'up KB/f':>9} {'down KB/f':>10} {'total KB/f':>11} {'fps':>6}")
    lines = []
    for form in forms:
        for bs in batch_sizes:
            batch = _make_batch(bs, rows, form)
            times, down, up = [], 0, 0
            for _ in range(reps):
                req = client._build_request(0, bs, batch, encoding=encoding,
                                            quality=quality,
                                            feature_dtype="float16")
                up = sum(len(t.data) for t in req.audiofeat) + len(req.feat_block.data)
                t0 = time.perf_counter()
                try:
                    reply = client._stub.InferBatch(req, timeout=180)
                except grpc.RpcError as exc:
                    detail = exc.details() or ""
                    if "no model loaded" in detail:
                        print("  worker has no backend loaded; only the UPLOAD band "
                              "was exercised. Connect an avatar for GPU timing.")
                        return lines, "no-model"
                    raise
                times.append(time.perf_counter() - t0)
                down = (sum(len(b) for b in reply.jpeg)
                        if reply.encoding == "jpeg-bgr" else len(reply.frames.data))
            best = min(times)
            per_frame = best / bs
            print(f"{form:>7} {bs:3d} {best*1000:9.1f} {per_frame*1000:9.1f} "
                  f"{up/bs/1024:9.1f} {down/bs/1024:10.1f} "
                  f"{(up+down)/bs/1024:11.1f} {1/per_frame:6.2f}")
            lines.append((form, bs, best, up / bs, down / bs))
    return lines, None


def fit(lines):
    """Fixed cost per batch vs per-frame slope, per form, from the extremes."""
    for form in sorted({line[0] for line in lines}):
        pts = sorted(((bs, t) for f, bs, t, _u, _d in lines if f == form))
        if len(pts) < 2:
            continue
        (bs_lo, t_lo), (bs_hi, t_hi) = pts[0], pts[-1]
        slope = (t_hi - t_lo) / (bs_hi - bs_lo)
        fixed = t_lo - slope
        print(f"fit {form:>7} over bs{bs_lo}..bs{bs_hi}: "
              f"fixed={fixed*1000:6.1f} ms/batch, slope={slope*1000:6.1f} ms/frame")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--address", default="")
    p.add_argument("--batch-sizes", default="4,8,16")
    p.add_argument("--formats", default="legacy,block",
                   help="legacy | block (comma separated)")
    p.add_argument("--rows", type=int, default=10,
                   help="feature rows per frame (the real pipeline uses 10)")
    p.add_argument("--encoding", default="jpeg-bgr")
    p.add_argument("--quality", type=int, default=75)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args(argv)

    from remote.client import RemoteClient, PipelinedInfer

    address = a.address or _address()
    client = RemoteClient(address, timeout=180)
    try:
        health = client.health()
    except Exception as exc:  # noqa: BLE001
        print(f"worker unreachable: {exc}", file=sys.stderr)
        print("check the tunnel first: python -m remote.tunnel status, "
              "scripts/check_ssh_forwarding.sh", file=sys.stderr)
        return 2
    print(f"health: device={health['device']} gpu={health['detail']} "
          f"backend={health['backend'] or 'none'} "
          f"avatar={health['avatar_id'] or 'none'} busy={health['busy']} "
          f"revision={health['revision']}")
    if health["busy"]:
        print("worker reports busy; numbers include someone else's work.",
              file=sys.stderr)

    sizes = [int(s) for s in a.batch_sizes.split(",") if s.strip()]
    forms = [f.strip() for f in a.formats.split(",") if f.strip()]
    lines, reason = measure(client, sizes, forms, a.reps, a.rows,
                            a.encoding, a.quality)
    fit(lines)

    legacy = [l for l in lines if l[0] == "legacy"]
    block = [l for l in lines if l[0] == "block"]
    if legacy and block:
        lu = sum(l[3] for l in legacy) / len(legacy)
        bu = sum(l[3] for l in block) / len(block)
        print(f"\nupload per frame: legacy {lu/1024:.1f} KB -> "
              f"block {bu/1024:.1f} KB  ({lu/bu:.2f}x smaller)")

    if reason == "no-model":
        return 3

    # Overlap check: two batches in flight should beat two sequential RPCs.
    bs = sizes[len(sizes) // 2]
    batch = _make_batch(bs, a.rows, "block")
    t0 = time.perf_counter()
    for i in range(2):
        client.infer_batch(i, bs, batch, encoding=a.encoding, quality=a.quality)
    sequential = time.perf_counter() - t0

    with PipelinedInfer(client, depth=2) as pipe:
        t0 = time.perf_counter()
        for i in range(2):
            pipe.run(i, bs, batch, encoding=a.encoding, quality=a.quality)
        piped = time.perf_counter() - t0

    print(f"\n2x{bs} frames: sequential {sequential*1000:.0f} ms "
          f"({2*bs/sequential:.2f} fps) -> pipelined {piped*1000:.0f} ms "
          f"({2*bs/piped:.2f} fps) = {sequential/piped:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
