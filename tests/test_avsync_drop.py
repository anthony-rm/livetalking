"""LT_AVSYNC=drop: speech paces at ARRIVAL rate; display shows the newest
rendered frame.

The stretch behaviour (audio coupled to rendered frames) stretched every
utterance at sub-25fps render rates. Drop mode taps real audio at
BaseASR.put_audio_frame into asr.audio_tap_queue and the pacer drains the
tap at 50 Hz — speech advances at wall rate no matter the render fps. These
tests are GPU-free and drive _pace_audio / process_frames with stubs the
way test_audio_pacing.py does.
"""
from __future__ import annotations

import os
import queue as _queue
import threading
import time
from queue import Queue

import numpy as np
import pytest


def _make_stub(lead_chunks=0):
    """A BaseAvatar spare enough for _pace_audio / process_frames, no init."""
    from avatars.base_avatar import BaseAvatar

    stub = BaseAvatar.__new__(BaseAvatar)
    stub.output = _Out()
    stub.asr = _AsrStub(lead_chunks)
    stub.audio_out_queue = Queue(maxsize=200)
    stub.res_frame_queue = Queue(maxsize=10)
    stub.frame_list_cycle = [np.zeros((16, 16, 3), np.uint8)]
    stub.speaking = False
    return stub


class _Out:
    def __init__(self):
        self.audio = []
        self.video = []
        self.started = False

    def push_audio_frame(self, chunk, userdata=None):
        self.audio.append((np.asarray(chunk).copy(), userdata))

    def push_video_frame(self, frame):
        self.video.append(frame)

    def start(self):
        self.started = True

    def stop(self):
        pass


class _AsrStub:
    chunk = 320
    audio_lead_chunks = 10

    def __init__(self, lead):
        self.audio_lead_chunks = lead
        self.feat_queue = Queue()
        self.audio_tap_queue = Queue(maxsize=400)

    def flush_talk(self):
        # Mirror BaseASR.flush_talk: main queue + tap queue.
        self.audio_tap_queue.queue.clear()


def _run_pacer(stub, quit_event, duration=0.6):
    t = threading.Thread(target=stub._pace_audio, args=(quit_event,),
                         daemon=True)
    t.start()
    time.sleep(duration)
    quit_event.set()
    t.join(timeout=3)
    return stub.output.audio


def _speech(n, value=1.0):
    # Keep magnitude small: _pace_audio scales by 32767 into int16, and a
    # value >1 clips to 32767 destroying the position encoding. ±0.001*i
    # stays far below clipping and survives the int16 cast.
    return np.full(320, value * 0.001, np.float32)


# ─── tap + pacer: speech at arrival rate ────────────────────────────────

def test_drop_mode_paces_speech_at_arrival_rate(monkeypatch):
    """20 bursts of speech arrive in ~40 ms; the pacer must still push ~1 per
    20 ms tick — and every REAL chunk eventually reaches the track in order."""
    monkeypatch.setenv("LT_AVSYNC", "drop")
    stub = _make_stub(lead_chunks=0)
    quit_event = threading.Event()

    chunks = [_speech(i, float(i + 1)) for i in range(20)]
    for c in chunks:
        stub.asr.audio_tap_queue.put(_frame(c))
    out = _run_pacer(stub, quit_event, duration=0.5)

    # In 0.5 s at 50 Hz ≈ 25 ticks: all 20 speech chunks + ~5 silence pad.
    real = [a for a, u in out if a.any()]
    assert len(real) >= 20, f"only {len(real)} of 20 speech chunks in 0.5 s"
    # Order preserved (value encodes position).
    vals = [float(np.max(a)) for a in real]
    assert vals == sorted(vals)
    # And the pacer never emitted a zero where speech was queued: positions
    # are 1..20 scaled by 0.001*32767 ≈ 32.767.
    assert abs(vals[0] - 32.767) < 1.0, vals[:3]


def test_drop_mode_fills_gaps_with_silence(monkeypatch):
    monkeypatch.setenv("LT_AVSYNC", "drop")
    stub = _make_stub(lead_chunks=0)
    quit_event = threading.Event()
    stub.asr.audio_tap_queue.put(_frame(_speech(1)))
    out = _run_pacer(stub, quit_event, duration=0.3)
    silent = [a for a, _ in out if not a.any()]
    assert len(silent) > 0
    # The exact tick count is load-flaky on this machine (5-15 observed in
    # a nominal 0.3 s window); the 50 Hz rate guarantee itself is covered by
    # test_audio_pacing.py. Bound only against runaway behavior.
    assert len(out) <= 20


def test_end_marker_plays_as_silence_but_keeps_eventpoint(monkeypatch):
    """TTS adapters end each utterance with an all-zero chunk carrying the
    'end' eventpoint. It must reach the track (silent) with its userdata."""
    monkeypatch.setenv("LT_AVSYNC", "drop")
    stub = _make_stub(lead_chunks=0)
    quit_event = threading.Event()
    stub.asr.audio_tap_queue.put(_frame(_speech(1), {"status": "start"}))
    stub.asr.audio_tap_queue.put(
        _frame(np.zeros(320, np.float32), {"status": "end", "text": "hi"}))
    out = _run_pacer(stub, quit_event, duration=0.2)
    events = [u for _, u in out if u]
    assert any(u and u.get("status") == "end" for u in events)


def test_barge_in_kills_delay_line_zombies(monkeypatch):
    """flush_talk clears the tap AND the pacer's delay line: chunks of the
    interrupted sentence that were already inside the line must not play."""
    monkeypatch.setenv("LT_AVSYNC", "drop")
    stub = _make_stub(lead_chunks=10)
    quit_event = threading.Event()

    # Pollute the line: 5 speech chunks, then barge-in, then NEW speech.
    # Values: old sentence 100-i (scaled 0.001 -> ~3.2-3.3 in int16),
    # new speech 900 (scaled ~29.5).
    for i in range(5):
        stub.asr.audio_tap_queue.put(_frame(_speech(2, 100.0 + i)))
    t = threading.Thread(target=stub._pace_audio, args=(quit_event,),
                         daemon=True)
    t.start()
    time.sleep(0.15)          # pacer pulled some/all into the delay line
    stub.flush_talk()          # barge-in: purges queues, bumps generation
    # A real sentence is many chunks: the delay line refills to lead+1
    # before playback resumes (identical to stretch-mode startup).
    for i in range(12):
        stub.asr.audio_tap_queue.put(_frame(_speech(3, 900.0)))
    time.sleep(0.8)
    quit_event.set()
    t.join(timeout=3)

    out = stub.output.audio
    # Old-sentence markers land at ~ (100..105)*0.001*32767 ≈ 3276-3292 —
    # any output sample in [3200, 3350] is zombie speech.
    markers = [a for a, _ in out
               if 3200.0 <= float(np.max(a)) <= 3350.0]
    after = [a for a, _ in out if float(np.max(a)) >= 29000.0]
    assert not markers, "zombie speech from the interrupted sentence played"
    assert after, "new speech after barge-in did not play"


def test_stretch_mode_reads_audio_out_queue(monkeypatch):
    monkeypatch.setenv("LT_AVSYNC", "stretch")
    stub = _make_stub(lead_chunks=0)
    quit_event = threading.Event()
    stub.audio_out_queue.put((np.full(320, 7, np.int16), None))
    out = _run_pacer(stub, quit_event, duration=0.2)
    assert any(float(np.max(a)) == 7.0 for a, _ in out)


# ─── process_frames: newest-frame display ───────────────────────────────

def _frame(data, userdata=None):
    from avatars.base_avatar import AudioFrameData

    return AudioFrameData(data=data, type=0, userdata=userdata or {})


def _silent_window():
    from avatars.base_avatar import AudioFrameData

    z = np.zeros(320, np.float32)
    return [AudioFrameData(data=z, type=1, userdata={}),
            AudioFrameData(data=z, type=1, userdata={})]


def _speech_window(value):
    from avatars.base_avatar import AudioFrameData

    d = np.full(320, value, np.float32)
    return [AudioFrameData(data=d, type=0, userdata={}),
            AudioFrameData(data=d, type=0, userdata={})]


def _run_process(stub, quit_event, items, duration=0.4):
    for item in items:
        stub.res_frame_queue.put(item)
    t = threading.Thread(target=stub.process_frames, args=(quit_event,),
                         daemon=True)
    t.start()
    time.sleep(duration)
    quit_event.set()
    t.join(timeout=3)


def test_no_speaking_frame_is_skipped_in_drop_mode(monkeypatch):
    """Drop mode must NOT drop rendered frames: the render queue holds a whole
    10-frame batch, so a queue-depth "staleness" check would discard 9 of
    every 10 frames and starve the video track (measured). The WebRTC
    track's newest-frame-wins decides what the browser shows."""
    monkeypatch.setenv("LT_AVSYNC", "drop")
    stub = _make_stub()
    stub.paste_back_frame = lambda res, idx: np.full(
        (16, 16, 3), int(float(np.max(res))), np.uint8)
    quit_event = threading.Event()
    old = np.full((2, 2), 1.0, np.float32)
    new = np.full((2, 2), 9.0, np.float32)
    _run_process(stub, quit_event, [
        (old, _speech_window(1.0), 0),
        (new, _speech_window(9.0), 0),
    ])
    assert len(stub.output.video) == 2, "a rendered frame was dropped"


def test_silent_frames_display_normally_in_drop_mode(monkeypatch):
    monkeypatch.setenv("LT_AVSYNC", "drop")
    stub = _make_stub()
    quit_event = threading.Event()
    z = np.zeros((2, 2), np.float32)
    _run_process(stub, quit_event, [
        (z, _silent_window(), 0),
        (z, _silent_window(), 0),
    ])
    assert len(stub.output.video) == 2


def test_stretch_mode_never_skips_speaking_frames(monkeypatch):
    monkeypatch.setenv("LT_AVSYNC", "stretch")
    stub = _make_stub()
    stub.paste_back_frame = lambda res, idx: np.zeros((16, 16, 3), np.uint8)
    quit_event = threading.Event()
    old = np.full((2, 2), 1.0, np.float32)
    new = np.full((2, 2), 9.0, np.float32)
    _run_process(stub, quit_event, [
        (old, _speech_window(1.0), 0),
        (new, _speech_window(9.0), 0),
    ])
    assert len(stub.output.video) == 2


def test_audio_out_queue_not_fed_in_drop_mode(monkeypatch):
    """Drop mode: process_frames must NOT forward chunks to audio_out_queue
    (the pacer reads the tap); stretch mode: it must."""
    monkeypatch.setenv("LT_AVSYNC", "drop")
    stub = _make_stub()
    stub.paste_back_frame = lambda res, idx: np.zeros((16, 16, 3), np.uint8)
    quit_event = threading.Event()
    _run_process(stub, quit_event, [
        (np.full((2, 2), 1.0, np.float32), _speech_window(1.0), 0),
    ])
    assert stub.audio_out_queue.qsize() == 0


# ─── stale idle silence ahead of speech ────────────────────────────────

def _idle_chunks(n):
    from avatars.base_avatar import AudioFrameData
    z = np.zeros(320, np.float32)
    return [AudioFrameData(data=z, type=1, userdata={}) for _ in range(n)]


def _asr_with_output_queue(tmp_path=None):
    """A BaseASR with a real output_queue, no model work."""
    import os, sys, types
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from avatars.audio_features.base_asr import BaseASR

    opt = types.SimpleNamespace(fps=25, batch_size=10, l=10, r=10)
    asr = BaseASR.__new__(BaseASR)
    asr.opt = opt
    asr.batch_size = opt.batch_size
    asr.chunk = 320
    asr.queue = Queue()
    asr.output_queue = Queue(maxsize=opt.batch_size * 4)
    asr.audio_tap_queue = Queue()
    asr._tap_seq = 0
    return asr


def test_stale_idle_silence_is_dropped_on_speech(monkeypatch):
    """Idle floods output_queue with type=1 silence; the first real chunk
    must clear it so the picture is not a source frame over live speech.

    Measured live before the fix: 19-20 consecutive static frames lasting
    ~1.5 s at the start of every sentence.
    """
    monkeypatch.setenv("LT_AVSYNC", "drop")
    asr = _asr_with_output_queue()
    for f in _idle_chunks(40):
        asr.output_queue.put(f)

    asr.put_audio_frame(np.full(320, 0.5, np.float32), {})

    assert asr.output_queue.qsize() == 0, "idle silence was not cleared"


def test_stale_idle_drop_keeps_real_audio(monkeypatch):
    """Any real chunk already queued stays, in order — never dropped."""
    monkeypatch.setenv("LT_AVSYNC", "drop")
    asr = _asr_with_output_queue()
    from avatars.base_avatar import AudioFrameData

    asr.output_queue.put(_idle_chunks(1)[0])
    r1 = AudioFrameData(data=np.full(320, 0.1, np.float32), type=0, userdata={})
    r2 = AudioFrameData(data=np.full(320, 0.2, np.float32), type=0, userdata={})
    asr.output_queue.put(r1)
    asr.output_queue.put(_idle_chunks(1)[0])
    asr.output_queue.put(r2)

    asr.put_audio_frame(np.full(320, 0.9, np.float32), {})

    kinds = []
    while not asr.output_queue.empty():
        kinds.append(asr.output_queue.get_nowait().type)
    assert kinds == [0, 0], kinds


def test_stretch_mode_keeps_idle_silence(monkeypatch):
    """Stretch mode paces audio against the picture, so the backlog is not
    visible there and must be left alone (old behaviour)."""
    monkeypatch.setenv("LT_AVSYNC", "stretch")
    asr = _asr_with_output_queue()
    for f in _idle_chunks(5):
        asr.output_queue.put(f)

    asr.put_audio_frame(np.full(320, 0.5, np.float32), {})

    assert asr.output_queue.qsize() == 5, "stretch mode must not purge"
