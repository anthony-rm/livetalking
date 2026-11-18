"""The audio pacer holds back `asr.audio_lead_chunks` chunks of every stream.

Why: the ASR pairs each feature batch with the audio sitting RIGHT-CONTEXT
ahead of it (the HuBERT window needs ~0.2 s of audio past a row's content
before the row exists). Without the hold, audio plays ~0.2 s before the lips
move — measured live as a -0.15..-0.40 s persistent v-a offset.
"""
import sys
import threading
import time
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, ".")


class _Out:
    def __init__(self):
        self.audio = []

    def push_audio_frame(self, chunk, userdata):
        self.audio.append((chunk.copy(), userdata))

    def push_video_frame(self, frame):
        pass


class _Asr:
    chunk = 320

    def __init__(self, lead):
        self.audio_lead_chunks = lead
        self.feat_queue = Queue()
        # Stretch-lane stub: these tests pin LT_AVSYNC=stretch (see
        # _run_pacer), so the tap queue below is never read — but providing
        # it keeps the stub shape honest if the default ever flips back.
        self.audio_tap_queue = Queue(maxsize=10)


def _run_pacer(lead, chunks, realtime=True, timeout=10.0):
    """Run BaseAvatar._pace_audio until `chunks` have been pushed out."""
    import os

    from avatars.base_avatar import BaseAvatar

    # This file exercises the STRETCH lane (audio_out_queue source); pin
    # the env so the drop-lane default cannot read a missing tap queue.
    os.environ["LT_AVSYNC"] = "stretch"

    av = object.__new__(BaseAvatar)
    av.__dict__["asr"] = _Asr(lead)
    out = _Out()
    av.__dict__["audio_out_queue"] = Queue()
    av.__dict__["output"] = out
    quit_event = threading.Event()

    for i, c in enumerate(chunks):
        av.audio_out_queue.put((c, {"i": i}))

    # the pacer pushes at 50 Hz; let it run until all chunks consumed (plus
    # the lead) or timeout
    t = threading.Thread(
        target=BaseAvatar._pace_audio, args=(av, quit_event), daemon=True)
    t.start()

    want = len(chunks) + lead
    t0 = time.perf_counter()
    while len(out.audio) < want and time.perf_counter() - t0 < timeout:
        time.sleep(0.01)
    quit_event.set()
    t.join(timeout=2)

    real = [c for c, ud in out.audio if (ud or {}).get("i") is not None]
    sil = len(out.audio) - len(real)
    return out.audio, real, sil


class AudioLeadHold(unittest.TestCase):
    def test_lead_delays_stream_in_order(self):
        chunks = [np.full(320, i + 1, np.int16) for i in range(30)]
        audio, real, sil = _run_pacer(lead=10, chunks=chunks)
        # the first >= lead emissions are silence, then the chunks in order;
        # extra trailing silence just means the queue ran dry before the
        # harness stopped the thread
        self.assertGreaterEqual(sil, 10)
        self.assertEqual([int(c[0]) for c in real], list(range(1, 31)))

    def test_zero_lead_is_passthrough(self):
        chunks = [np.full(320, i + 1, np.int16) for i in range(20)]
        audio, real, sil = _run_pacer(lead=0, chunks=chunks)
        self.assertEqual([int(c[0]) for c in real], list(range(1, 21)))
        # no lead means every silence in the output comes from a dry queue
        # AFTER all real chunks; none may precede them
        ud_i = [(ud or {}).get("i") for _, ud in audio]
        first_real = ud_i.index(0)
        self.assertEqual(first_real, 0)


if __name__ == "__main__":
    unittest.main()
