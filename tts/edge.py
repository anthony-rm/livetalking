import time
import asyncio
import os
import socket
import numpy as np
import resampy
import soundfile as sf
import edge_tts
from io import BytesIO

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


@register("tts", "edgetts")
class EdgeTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        text,textevent = msg
        voice = self.opt.REF_FILE or "zh-CN-YunxiaNeural" 
        voicename = textevent.get('tts', {}).get('ref_file',voice) #self.opt.REF_FILE #"zh-CN-YunxiaNeural"
        t = time.time()
        asyncio.new_event_loop().run_until_complete(self.__main(voicename,text))
        logger.info(f'-------edge tts time:{time.time()-t:.4f}s')
        if self.input_stream.getbuffer().nbytes<=0: #edgetts err
            logger.error('edgetts err!!!!!')
            return
        
        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            eventpoint={}
            streamlen -= self.chunk
            if idx==0:
                eventpoint={'status':'start','text':text}
            elif streamlen<self.chunk:
                eventpoint={'status':'end','text':text}
            eventpoint.update(**textevent) #eventpoint={'status':'end','text':text,'msgevent':textevent}
            self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
            idx += self.chunk
        #if streamlen>0:  #skip last frame(not 20ms)
        #    self.queue.put(stream[idx:])
        self.input_stream.seek(0)
        self.input_stream.truncate() 

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
    async def __main(self,voicename: str, text: str):
        try:
            _install_dns_fallback()
            communicate = edge_tts.Communicate(text, voicename)

            #with open(OUTPUT_FILE, "wb") as file:
            first = True
            async for chunk in communicate.stream():
                if first:
                    first = False
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    #self.push_audio(chunk["data"])
                    self.input_stream.write(chunk["data"])
                    #file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            logger.exception('edgetts')
