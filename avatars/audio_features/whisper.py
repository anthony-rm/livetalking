###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  Whisper 音频特征提取 — 用于 MuseTalk
#  迁移自 museasr.py
#

import time
import numpy as np

import queue
from queue import Queue
from avatars.audio_features.base_asr import BaseASR
from avatars.musetalk.whisper.audio2feature import Audio2Feature
from utils.logger import logger


class FeatBatch(list):
    """A batch of whisper chunks that also knows how to ship itself compactly.

    It IS a list, so every existing consumer keeps working unchanged (indexing,
    iteration, ``np.stack(batch)``, ``MelASR``-style lists). The three extra
    attributes describe the same batch as ONE contiguous slice of the feature
    array, which is how it can travel 3.3x smaller -- see
    ``WhisperASR._attach_block``. When ``block`` is None the batch has no
    compact form and the client must send the legacy per-frame tensors.
    """

    __slots__ = ("block", "offsets", "rows_per_frame", "why_not")

    def __init__(self, chunks):
        super().__init__(chunks)
        self.block = None
        self.offsets = ()
        self.rows_per_frame = 0
        self.why_not = ""


class WhisperASR(BaseASR):
    def __init__(self, opt, parent, audio_processor:Audio2Feature):
        super().__init__(opt, parent)
        self.audio_processor = audio_processor
    
    def _feature2chunks(self,feature_array,batch_size,audio_feat_win=[8,8],start=0,feature_idx_multiplier=1.0):
        """
        :param feature_array: 
        :param batch_size: batch大小
        :param audio_feat_win: 音频特征窗口大小，通常为 [左侧窗口大小, 右侧窗口大小]，单位为视频帧数
        :param start: 起始帧索引，通常为 stride_left_size/2
        :param feature_idx_multiplier: 用于将视频帧索引转换为特征索引的乘数，通常为 (特征提取的宽度 / 视频帧率)
        :return: 
        """
        feature_chunks = []
        #start += 10
        #feature_idx_multiplier = 50./fps 
        for i in range(batch_size):
            # start_idx = int(i * whisper_idx_multiplier)
            # if start_idx>=len(feature_array):
            #     break
            selected_feature,selected_idx = self._get_sliced_feature(
                feature_array=feature_array, vid_idx=i+start,
                audio_feat_win=audio_feat_win, feature_idx_multiplier=feature_idx_multiplier)
            #print(f"i:{i},selected_idx {selected_idx},feature_idx_multiplier:{feature_idx_multiplier}")
            feature_chunks.append(selected_feature.reshape(-1, 384))
        return feature_chunks

    def run_step(self):
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        for _ in range(self.batch_size*2):
            audio_frame = self.get_audio_frame()
            self.frames.append(audio_frame.data)
            self.output_queue.put(audio_frame)
        
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        inputs = np.concatenate(self.frames) # [N * chunk]
        whisper_feature = self.audio_processor.audio2feat(inputs)
        self._last_feature = whisper_feature
        whisper_chunks = self._feature2chunks(feature_array=whisper_feature,batch_size=self.batch_size,
                                              audio_feat_win = [2,3],start=self.stride_left_size/2,
                                              feature_idx_multiplier=2)
        self.feat_queue.put(self._attach_block(whisper_feature, whisper_chunks,
                                              [2, 3], self.stride_left_size/2, 2))
        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

    def _attach_block(self, feature_array, chunks, audio_feat_win, start,
                      feature_idx_multiplier):
        """Return a FeatBatch carrying a compact wire form when one exists.

        Always returns a usable list of chunks; the compact form is a bonus. A
        refusal is logged once per session (not per batch) because it is a
        throughput regression the user needs to see, not a correctness event.
        """
        batch = FeatBatch(chunks)
        rows_per_frame = None
        if chunks:
            flat = chunks[0].size // chunks[0].shape[-1]
            rows_per_frame = flat // int(np.prod(feature_array.shape[1:-1]))
        try:
            spans = self._chunk_spans(feature_array, len(chunks), audio_feat_win,
                                      start, feature_idx_multiplier)
            block, offsets, rows, why_not = self.feature_block(
                feature_array, chunks, spans, rows_per_frame)
        except Exception as exc:  # noqa: BLE001
            block, offsets, rows, why_not = None, (), 0, f"error: {exc}"
        if block is None:
            if not getattr(self, "_block_refusal_logged", False):
                logger.warning(
                    "[asr] compact audio-feature block unavailable (%s); "
                    "sending per-frame tensors -- the upload will be ~3x larger "
                    "and fps will be link-bound", why_not)
                self._block_refusal_logged = True
            batch.why_not = why_not
            return batch
        batch.block = block
        batch.offsets = offsets
        batch.rows_per_frame = rows
        return batch

    # ── wire-size reduction ────────────────────────────────────────────────
    # The chunks above are OVERLAPPING slices of ONE feature array: with the
    # window/stride used here, frame i reads rows [2i+6, 2i+16), so consecutive
    # frames share 8 of their 10 rows. Shipping one tensor per frame therefore
    # re-sends the same bytes n times (80 rows for a batch of 8 that needs 24).
    #
    # We do NOT send only the "new" rows per frame and reassemble across steps:
    # whisper's encoder is context-dependent, so the same audio time produces
    # DIFFERENT features when the input window slides (measured: 9-40% relative
    # difference). Only the WITHIN-batch overlap is byte-exact, so that alone is
    # what gets deduplicated.
    def _chunk_spans(self, feature_array, batch_size, audio_feat_win=[2, 3],
                     start=0, feature_idx_multiplier=1.0):
        """Per-frame (begin, end) row ranges, from the SAME slicing the chunks use.

        ``_get_sliced_feature`` returns the clamped row indices it actually
        read, and those are monotonic non-decreasing, so [idx[0], idx[-1]+1) is
        a contiguous range that reproduces the chunk exactly.
        """
        spans = []
        for i in range(batch_size):
            _, idx = self._get_sliced_feature(
                feature_array=feature_array, vid_idx=i + start,
                audio_feat_win=audio_feat_win,
                feature_idx_multiplier=feature_idx_multiplier)
            spans.append((int(idx[0]), int(idx[-1]) + 1))
        return spans

    def feature_block(self, feature_array, chunks, spans, rows_per_frame):
        """Return (block, offsets, rows, why_not) covering the batch's rows.

        ``block`` is the slice of the feature array that touches every frame in
        the batch and ``offsets[i]`` is frame i's start row inside it, so the
        worker can rebuild frame i as ``block[off:off+rows]``. That
        reconstruction is PROVEN byte-identical here, per batch, against the
        chunks actually being sent -- never assumed. If it does not hold (a
        different window shape, a clamping edge, a future backend), the compact
        form is refused and ``why_not`` says so, so the caller falls back to
        legacy tensors and the log explains why throughput did not improve.
        """
        if not spans or not chunks or rows_per_frame <= 0:
            return None, (), 0, "no spans/chunks"
        if any((end - begin) != rows_per_frame for begin, end in spans):
            return None, (), 0, "ragged frame windows"
        lo = min(begin for begin, _ in spans)
        hi = max(end for _, end in spans)
        if lo < 0 or hi > len(feature_array):
            return None, (), 0, "window clamped at the array edge"
        # Skip when it does not actually save: a single-frame or
        # non-overlapping batch gains nothing by taking this path.
        per_frame_rows = sum(end - begin for begin, end in spans)
        if (hi - lo) * 3 > per_frame_rows * 2:      # <33% saving
            return None, (), 0, "saving too small (<33%)"

        block = feature_array[lo:hi]
        offsets = [begin - lo for begin, _ in spans]
        # Prove it, do not assume it: the chunk is the row range reshaped, and
        # the worker will rebuild exactly that. A mismatch here would be a
        # silent lip-sync corruption, so it is a hard reject.
        for i, chunk in enumerate(chunks):
            window = block[offsets[i]:offsets[i] + rows_per_frame]
            if np.asarray(window).reshape(chunk.shape).shape != chunk.shape:
                return None, (), 0, f"shape mismatch on frame {i}"
            if not np.array_equal(np.asarray(window).reshape(chunk.shape), chunk):
                return None, (), 0, f"values differ on frame {i}"
        return block, tuple(offsets), rows_per_frame, ""



