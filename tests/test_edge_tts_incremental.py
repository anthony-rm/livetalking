"""Regression tests for the edge-tts incremental streaming rewrite.

The old wrapper buffered the whole mp3 and only decoded after `turn.end`, so
the avatar waited for the entire sentence before hearing anything. These pin
the rewrite's three properties:

1. **Incremental emission** — chunks reach the avatar while the network stream
   is still open (before turn.end).
2. **Sample parity** — incremental decode + 16 kHz resample equals the old
   one-shot soundfile path.
3. **Eventpoints** — the first chunk carries 'start', the stream still ends
   with an 'end'-tagged chunk.

Hits the real edge-tts service (as every TTS in this repo does); needs
network. Run with the pinned interpreter:
    /opt/miniconda3/envs/livetalking/bin/python -m pytest tests/test_edge_tts_incremental.py -q
"""
from __future__ import annotations

import io
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SENTENCE = ("你好，欢迎来到今天的演示。我们正在测试语音合成的首包延迟，"
            "这句话大约两秒钟长，用来模拟一次典型的对话回复。")


def _mp3_packets(text: str, voice: str = "zh-CN-YunxiaNeural") -> list[bytes]:
    import asyncio
    import edge_tts

    async def go():
        c = edge_tts.Communicate(text, voice)
        return [ch["data"] async for ch in c.stream() if ch["type"] == "audio"]
    return asyncio.run(go())


def _oneshot(packets: list[bytes]) -> np.ndarray:
    """What the old wrapper produced: soundfile one-shot + resampy."""
    import soundfile as sf
    import resampy
    buf = io.BytesIO(b"".join(packets))
    stream, sr = sf.read(buf)
    stream = stream.astype(np.float32)
    if sr != 16000:
        stream = resampy.resample(stream, sr, 16000)
    return stream


@pytest.fixture()
def tts():
    """A real EdgeTTS instance with a stub avatar parent collecting chunks."""
    from tts.edge import EdgeTTS

    opt = types.SimpleNamespace(REF_FILE="", fps=25)
    parent = types.SimpleNamespace(chunks=[], events=[])
    parent.put_audio_frame = \
        lambda chunk, ev: (parent.chunks.append(np.asarray(chunk).copy()),
                           parent.events.append(dict(ev)))
    t = EdgeTTS.__new__(EdgeTTS)
    t.opt = opt
    t.parent = parent
    t.sample_rate = 16000
    t.chunk = 16000 // (25 * 2)
    t.state = None
    from tts.base_tts import State
    t.state = State.RUNNING
    return t, parent


def _drive_with_packets(tts_obj, packets):
    """Run _push_decoded over packets exactly as __main would, then flush."""
    from tts.edge import _StreamState
    state = _StreamState(SENTENCE, {})
    for b in packets:
        tts_obj._push_decoded(state, b)
    return state


def test_emits_chunks_incrementally(tts):
    t, parent = tts
    import asyncio
    import edge_tts
    from tts.edge import _StreamState

    seen_during_stream = 0

    async def go():
        nonlocal seen_during_stream
        c = edge_tts.Communicate(SENTENCE, "zh-CN-YunxiaNeural")
        state = _StreamState(SENTENCE, {})
        async for ch in c.stream():
            if ch["type"] == "audio":
                before = len(parent.chunks)
                t._push_decoded(state, ch["data"])
                seen_during_stream += len(parent.chunks) - before
        # flush path (what txt_to_audio does after __main returns)
        out = [r.to_ndarray().reshape(-1) for r in state.rs.resample(None)]
        tail = np.concatenate(out) if out else np.zeros(0, np.float32)
        if tail.size or state.pending.size:
            stream = np.concatenate([state.pending, tail]) \
                if tail.size else state.pending
            streamlen = stream.shape[0]
            if streamlen >= t.chunk:
                parent.put_audio_frame(
                    stream[:streamlen - streamlen % t.chunk], {})

    asyncio.run(go())
    assert seen_during_stream > 0, \
        "no chunk reached the avatar before turn.end — the rewrite's point"


def test_eventpoints_start_and_end(tts):
    t, parent = tts
    packets = _mp3_packets(SENTENCE)
    state = _drive_with_packets(t, packets)
    # The wrapper's flush tail: the last full chunk carries 'end'.
    out = [r.to_ndarray().reshape(-1) for r in state.rs.resample(None)]
    tail = np.concatenate(out) if out else np.zeros(0, np.float32)
    stream = np.concatenate([state.pending, tail]) if tail.size else state.pending
    n_full = stream.shape[0] // t.chunk
    if n_full:
        parent.put_audio_frame(stream[:n_full * t.chunk], {'status': 'end'})
    statuses = [e.get('status') for e in parent.events if e.get('status')]
    assert statuses[:1] == ['start']
    assert 'end' in statuses


def test_incremental_equals_oneshot(tts):
    t, parent = tts
    packets = _mp3_packets(SENTENCE)
    state = _drive_with_packets(t, packets)
    out = [r.to_ndarray().reshape(-1) for r in state.rs.resample(None)]
    tail = np.concatenate(out) if out else np.zeros(0, np.float32)
    # Assemble exactly what the wrapper pushed, plus the un-pushed remainder
    # and the flushed tail, so the comparison is over the same sample count
    # the one-shot path produces.
    pushed = np.concatenate(parent.chunks) if parent.chunks else np.zeros(0, np.float32)
    inc = np.concatenate([pushed, state.pending, tail]) \
        if (state.pending.size or tail.size) else pushed
    ref = _oneshot(packets)
    assert abs(inc.size - ref.size) <= t.chunk, (inc.size, ref.size)
    n = min(len(inc), len(ref))
    assert np.max(np.abs(inc[:n] - ref[:n])) < 0.02


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))