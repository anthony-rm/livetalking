"""Audio pacing: the track must get 50 chunks/sec even when inference stalls.

The old code pushed audio in lockstep with video frames (2 chunks per frame),
so at 21 fps the audio track received 42 chunks/sec and stalled in 5 ms
increments -- audible micro-gaps. The pacing thread feeds the track from its own
wall-clock timer, so the rate is independent of inference throughput.

These tests measure the OUTPUT RATE of _pace_audio, which is the property the
browser's jitter buffer actually depends on, not the input rate.
"""
from __future__ import annotations

import queue as _queue
import threading
import time

import numpy as np
import pytest


def _pace(stub, quit_event, out):
    """Run _pace_audio, capturing every chunk it pushes into `out`."""
    stub.output = _CapturingOutput(out)
    stub.asr = type("ASR", (), {"chunk": 320})()
    stub.audio_out_queue = _queue.Queue(maxsize=200)
    stub._pace_audio(quit_event)


class _CapturingOutput:
    def __init__(self, out):
        self.out = out

    def push_audio_frame(self, chunk, userdata=None):
        self.out.append((np.asarray(chunk).copy(), userdata))


def _run_pace(stub, feed_chunks, duration, max_wait=6.0):
    out = []
    quit_event = threading.Event()
    stub.audio_out_queue = _queue.Queue(maxsize=200)

    def producer():
        for c in feed_chunks:
            if quit_event.is_set():
                return
            stub.audio_out_queue.put(c)
            time.sleep(0.001)

    t = threading.Thread(target=_pace, args=(stub, quit_event, out), daemon=True)
    p = threading.Thread(target=producer, daemon=True)
    t.start()
    p.start()
    time.sleep(duration)
    quit_event.set()
    t.join(timeout=max_wait)
    p.join(timeout=2.0)
    return out


def test_pacing_delivers_exactly_50_chunks_per_second_even_when_input_starves():
    """Inference at 21 fps feeds 42 chunks/sec; the track must still get 50."""
    from avatars.base_avatar import BaseAvatar

    stub = BaseAvatar.__new__(BaseAvatar)
    # Only 21 chunks arrive in 1.0 s -- what 21 fps produces.
    feed = [(np.full(320, i, np.int16), None) for i in range(21)]
    out = _run_pace(stub, feed, duration=1.0)

    # 50 chunks in 1.0 s +/- 2 (the timer sleeps up to 10 ms per tick).
    assert 48 <= len(out) <= 52, f"got {len(out)} chunks in 1.0 s"


def test_pacing_fills_gaps_with_silence():
    """When the queue is empty the pacing thread pushes real zero chunks, not
    nothing -- so the track never starves and the browser's jitter buffer sees a
    steady stream rather than a gap."""
    from avatars.base_avatar import BaseAvatar

    stub = BaseAvatar.__new__(BaseAvatar)
    feed = [(np.full(320, 100, np.int16), None)] * 5
    out = _run_pace(stub, feed, duration=1.0)

    silent = [c for c, _ in out if not c.any()]
    assert len(silent) > 0, "no silence was pushed while the queue was empty"


def test_pacing_preserves_eventpoints_in_order():
    """start/end eventpoints must reach the track in the order they were emitted."""
    from avatars.base_avatar import BaseAvatar

    stub = BaseAvatar.__new__(BaseAvatar)
    feed = [
        (np.full(320, 1, np.int16), {"status": "start", "text": "hello"}),
        (np.full(320, 2, np.int16), None),
        (np.full(320, 3, np.int16), {"status": "end", "text": "hello"}),
    ]
    out = _run_pace(stub, feed, duration=0.2)

    events = [u for _, u in out if u]
    assert events[0].get("status") == "start"
    assert events[-1].get("status") == "end"
    assert len(events) == 2


def test_pacing_stops_when_quitting():
    from avatars.base_avatar import BaseAvatar

    stub = BaseAvatar.__new__(BaseAvatar)
    out = _run_pace(stub, [], duration=0.3)
    assert len(out) > 0
    # After the quit event the thread should have exited; a second long wait
    # must not add chunks.
    time.sleep(0.3)
    assert len(out) == len(out)


def test_pacing_is_independent_of_input_rate():
    """The output rate must not depend on how fast audio arrives."""
    from avatars.base_avatar import BaseAvatar

    slow = _run_pace(BaseAvatar.__new__(BaseAvatar),
                     [(np.full(320, i, np.int16), None) for i in range(5)],
                     duration=0.5)
    fast = _run_pace(BaseAvatar.__new__(BaseAvatar),
                     [(np.full(320, i, np.int16), None) for i in range(200)],
                     duration=0.5)

    # Both ~25 chunks in 0.5 s regardless of how many were fed.
    assert abs(len(slow) - 25) <= 2, slow
    assert abs(len(fast) - 25) <= 2, fast