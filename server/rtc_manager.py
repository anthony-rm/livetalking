###############################################################################
#  WebRTC 连接管理 + RTC 音频/视频接收
###############################################################################

import json
import asyncio
import random
import copy
from typing import Dict, Optional
import queue

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from utils.logger import logger


# def _rand_session_id(n: int = 6) -> int:
#     """生成 N 位随机 session ID"""
#     return random.randint(10 ** (n - 1), 10 ** n - 1)


from server.session_manager import session_manager
from server.session_manager import MaxSessionError

class RTCManager:
    """
    WebRTC 连接管理器。
    
    管理 PeerConnection 生命周期、音视频轨道收发、DataChannel。
    """

    def __init__(self, opt):
        """
        Args:
            opt: 全局配置
        """
        self.opt = opt
        self.pcs: set = set()

    async def _create_pc_and_answer(self, avatar_session, sessionid, offer):
        """创建 PeerConnection、添加轨道、SDP 交换，返回已完成 answer 的 pc"""
        # stun 为空时传空列表（禁用 STUN），否则传入指定 STUN
        ice_servers = [RTCIceServer(urls=self.opt.stun)] if self.opt.stun else []
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ice_servers)
        )
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await pc.close()
                self.pcs.discard(pc)
                session_manager.remove_session(sessionid)

        # 添加发送轨道
        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        # 设置编解码器偏好
        #
        # H.264 需要 SDP 中携带 sprop-parameter-sets (SPS/PPS) 才能被解码，
        # 而 aiortc 不会生成该参数 —— Chrome 宽容可以接受，Firefox 则无法解码，
        # 表现为已连接但视频全黑。VP8 无需任何额外 SDP 参数，所有浏览器均可解码，
        # 因此默认优先使用 VP8。
        codec_order = [c.strip().upper() for c in
                       getattr(self.opt, 'video_codec', 'vp8').split(',') if c.strip()]
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = []
        for name in codec_order:
            for c in capabilities.codecs:
                # match on the codec family; keep every profile variant
                if c.name.upper() != name:
                    continue
                if c not in preferences:
                    preferences.append(c)
        # rtx 始终附加在末尾
        for c in capabilities.codecs:
            if c.name == "rtx" and c not in preferences:
                preferences.append(c)
        if preferences:
            # 视频 transceiver 是索引 1（索引 0 为音频）
            video_transceiver = next(
                (t for t in pc.getTransceivers() if t.kind == "video"), None)
            if video_transceiver is not None:
                video_transceiver.setCodecPreferences(preferences)
                logger.info("Video codec preference: %s",
                            ",".join(c.name for c in preferences))

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return pc

    async def handle_offer(self, request):
        """处理 WebRTC offer 信令"""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        try:
            sessionid = await session_manager.create_session(params)
        except MaxSessionError as e:
            logger.warning("Rejecting offer: %s", e)
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": str(e)}),
            )
        except Exception as e:
            # e.g. model lazy-load failure — return JSON so browser clients
            # surface a readable error instead of a JSON.parse exception.
            logger.exception("offer failed")
            return web.Response(
                status=503,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": str(e)}),
            )
        logger.info('offer sessionid=%s', sessionid)

        try:
            pc = await self._create_pc_and_answer(
                session_manager.get_session(sessionid), sessionid, offer
            )
        except Exception:
            logger.exception("offer SDP exchange failed")
            session_manager.remove_session(sessionid)
            return web.Response(
                status=503,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "SDP exchange failed"}),
            )

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid,
            }),
        )

    async def handle_whep(self, request):
        """
        处理 WHEP 信令（WebRTC HTTP Egress Protocol）

        - 请求 body 为裸 SDP offer（Content-Type: application/sdp）
        - 扩展参数通过 query string 传入（avatar, tts, tts_server 等）
        - 返回 SDP answer（Content-Type: application/sdp）
        - sessionid 通过 X-Session-ID 响应头返回
        """
        params = dict(request.query)
        # 客户端可通过 query param 自定义 sessionid，不传则自动生成
        client_sid = params.pop("sessionid", None)

        offer_sdp = await request.text()
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")

        try:
            sessionid = await session_manager.create_session(params, sessionid=client_sid)
        except MaxSessionError as e:
            logger.warning("Rejecting whep: %s", e)
            return web.Response(
                status=503,
                content_type="text/plain",
                text=str(e),
            )
        except Exception as e:
            logger.exception("whep failed")
            return web.Response(
                status=503,
                content_type="text/plain",
                text=str(e),
            )
        logger.info("whep sessionid=%s", sessionid)

        pc = await self._create_pc_and_answer(
            session_manager.get_session(sessionid), sessionid, offer
        )

        return web.Response(
            status=201,
            content_type="application/sdp",
            text=pc.localDescription.sdp,
            headers={"X-Session-ID": sessionid},
        )

    async def handle_rtcpush(self, push_url, sessionid: str):
        """RTCPush 模式：主动推流"""
        import aiohttp
        await session_manager.create_session({}, sessionid)
        avatar_session = session_manager.get_session(sessionid)

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        await pc.setLocalDescription(await pc.createOffer())

        async with aiohttp.ClientSession() as session:
            async with session.post(push_url, data=pc.localDescription.sdp) as response:
                answer_sdp = await response.text()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type='answer')
        )

    async def shutdown(self):
        """关闭所有 PeerConnection"""
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()
