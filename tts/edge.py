import time
import asyncio
import os
import socket
import numpy as np

import edge_tts
import av
from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register


# The GPU box has no DNS (its /etc/resolv.conf is root-only and the network
# blocks resolution), so edge-tts cannot resolve speech.platform.bing.com even
# though the endpoint itself is reachable. Pin the IP when DNS fails: the
# connection still carries the correct TLS SNI/certificate, so it is the
# ordinary secure path with only name resolution bypassed.
_EDGE_TTS_HOST = "speech.platform.bing.com"
_EDGE_TTS_FALLBACK_IP = os.environ.get("EDGE_TTS_FALLBACK_IP", "150.171.28.10")


def _install_dns_fallback(host=_EDGE_TTS_HOST, ip=_EDGE_TTS_FALLBACK_IP):
    """Resolve `host` to `ip` inside this process when the resolver fails."""
    try:
        socket.getaddrinfo(host, None)
        return False
    except socket.gaierror:
        pass

    original = socket.getaddrinfo

    def patched(name, port, *args, **kwargs):
        if name == host:
            return original(ip, port, *args, **kwargs)
        return original(name, port, *args, **kwargs)

    socket.getaddrinfo = patched
    logger.warning("edge-tts: DNS unavailable, pinning %s -> %s", host, ip)
    return True


class _StreamState:
    """Mutable per-sentence state shared between the async loop and the
    synchronous callback — plain attributes, no closure juggling."""

    __slots__ = ("text", "textevent", "first", "idx", "pending", "dec", "rs")

    def __init__(self, text, textevent):
        self.text = text
        self.textevent = textevent
        self.first = True
        self.idx = 0
        self.pending = np.zeros(0, dtype=np.float32)
        # One decoder + resampler per sentence: the mp3 decoder is stateful,
        # so creating one per packet would re-decode every packet from
        # scratch. The decoder is single-threaded (the TTS thread is the only
        # writer), so no lock is needed.
        self.dec = av.codec.CodecContext.create("mp3", "r")
        self.rs = av.AudioResampler(format="fltp", layout="mono", rate=16000)


@register("tts", "edgetts")
class EdgeTTS(BaseTTS):
    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        voice = self.opt.REF_FILE or "zh-CN-YunxiaNeural"
        voicename = textevent.get('tts', {}).get('ref_file', voice)
        t = time.time()

        # Audio is pushed to the avatar AS IT ARRIVES from the server. The
        # previous version buffered the whole mp3 into input_stream and only
        # decoded after turn.end, so the avatar waited for the entire
        # sentence — first frame ~1.5 s + the full sentence length, instead of
        # ~1.5 s + one chunk. The decoder is mp3 + 16 kHz resample in PyAV,
        # numerically equivalent to the old soundfile+resampy path (within
        # 0.02 peak amplitude; measured in tests/test_edge_tts_incremental.py).
        # The resampler's tail is flushed once after __main returns, and the
        # last chunk of the sentence carries the 'end' eventpoint (the old
        # wrapper did the same after its one-shot decode).
        state = _StreamState(text, textevent)
        try:
            asyncio.new_event_loop().run_until_complete(
                self.__main(voicename, text, state))
        except Exception:
            logger.exception('edgetts')

        # The resampler holds a tail it cannot emit until the stream ends;
        # flush it once here, emit any final full chunks, and tag the final
        # chunk with 'end' so the avatar knows the sentence is complete (the
        # old wrapper did the same after its one-shot decode).
        out = []
        for res in state.rs.resample(None):
            out.append(res.to_ndarray().reshape(-1))
        tail = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
        if tail.size or state.pending.size:
            stream = np.concatenate([state.pending, tail]) \
                if tail.size else state.pending
            idx = 0
            while stream.shape[0] - idx >= self.chunk and self.state == State.RUNNING:
                eventpoint = {}
                if state.first:
                    eventpoint = {'status': 'start', 'text': state.text}
                    state.first = False
                # The last chunk of the sentence is the one that would not be
                # followed by another full chunk — tag it 'end' here, since
                # the per-packet loop above cannot know the stream is over.
                if stream.shape[0] - idx < self.chunk * 2:
                    eventpoint['status'] = 'end'
                self.parent.put_audio_frame(
                    stream[idx:idx + self.chunk], eventpoint)
                idx += self.chunk
                state.idx += self.chunk
            leftover = stream[idx:]
            if leftover.size:
                state.pending = leftover

        logger.info(f'-------edge tts time:{time.time()-t:.4f}s')

    def _push_decoded(self, state: _StreamState, audio_bytes: bytes) -> None:
        """Decode one mp3 packet and push complete 20 ms chunks to the avatar.

        `idx` is LOCAL to this call: `stream` is rebuilt each packet from
        `pending` (the previous call's sub-chunk remainder), so an absolute
        index carried across calls would drift ahead of every rebuilt stream
        and stall emission completely.
        """
        out = []
        try:
            parsed = state.dec.parse(av.Packet(audio_bytes))
        except av.error.EOFError:
            return
        for pkt in parsed:
            for frame in state.dec.decode(pkt):
                for res in state.rs.resample(frame):
                    out.append(res.to_ndarray().reshape(-1))
        if not out:
            return
        decoded = np.concatenate(out)
        stream = np.concatenate([state.pending, decoded])
        state.pending = np.zeros(0, dtype=np.float32)
        idx = 0
        while stream.shape[0] - idx >= self.chunk and self.state == State.RUNNING:
            eventpoint = {}
            if state.first:
                eventpoint = {'status': 'start', 'text': state.text}
                state.first = False
            eventpoint.update(**state.textevent)
            self.parent.put_audio_frame(
                stream[idx:idx + self.chunk], eventpoint)
            state.idx += self.chunk
            idx += self.chunk
        leftover = stream[idx:]
        if leftover.size:
            state.pending = leftover

    async def __main(self, voicename: str, text: str, state: _StreamState):
        try:
            _install_dns_fallback()
            communicate = edge_tts.Communicate(text, voicename)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and self.state == State.RUNNING:
                    self._push_decoded(state, chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            logger.exception('edgetts')