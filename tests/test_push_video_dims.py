"""push_video must emit even dimensions for any --video-scale.

libx264 refuses odd heights ('height not divisible by 2') and the aiortc
video track then dies silently while audio keeps playing. 960-wide scaling
of a 1080x1920 frame yields 1707, which is exactly how that was found.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


class _Q:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _Track:
    def __init__(self):
        self._queue = _Q()


class _Container:
    def __init__(self, scale):
        self.opt = SimpleNamespace(video_scale=scale)
        self.output = None


class PushVideoEvenDims(unittest.TestCase):
    def _push(self, scale, frame):
        from server.webrtc import HumanPlayer

        player = object.__new__(HumanPlayer)
        player.__dict__["_HumanPlayer__video"] = _Track()
        player.__dict__["_HumanPlayer__container"] = _Container(scale)
        with mock.patch("av.VideoFrame") as vf:
            vf.from_ndarray.side_effect = lambda arr, format: arr
            player.push_video(frame)
        return player.__dict__["_HumanPlayer__video"]._queue.items[0][0]

    def test_960_of_1080x1920_is_even(self):
        out = self._push(960, np.zeros((1920, 1080, 3), np.uint8))
        self.assertEqual(out.shape, (1706, 960, 3))

    def test_720_of_1080x1920_unchanged(self):
        out = self._push(720, np.zeros((1920, 1080, 3), np.uint8))
        self.assertEqual(out.shape, (1280, 720, 3))

    def test_1080_no_resize(self):
        out = self._push(1080, np.zeros((1920, 1080, 3), np.uint8))
        # target_w == frame width -> no resize at all
        self.assertEqual(out.shape, (1920, 1080, 3))

    def test_no_scale_passthrough(self):
        out = self._push(0, np.zeros((1000, 500, 3), np.uint8))
        self.assertEqual(out.shape, (1000, 500, 3))


if __name__ == "__main__":
    unittest.main()
