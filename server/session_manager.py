###############################################################################
#  全局会话管理器 (Session Manager)
###############################################################################

import asyncio
import uuid
from typing import Dict, Optional
from utils.logger import logger
from avatars.base_avatar import BaseAvatar


class MaxSessionError(Exception):
    """会话数达到上限时抛出"""
    pass

class SessionKilledError(Exception):
    """会话在构建期间被管理员终止时抛出"""
    pass

def _rand_session_id() -> str:
    """生成 UUID session ID"""
    return str(uuid.uuid4())


def _release(avatar_session) -> None:
    """置位 quit_event，级联停掉 render/推理/合帧/TTS 线程。

    会话可能在 render() 之前就被丢弃 (构建期间被终止、SDP 交换失败)，此时
    还没有 quit_event —— 那种情况没有线程要停，丢掉引用即可。
    """
    if avatar_session is None:
        return
    if getattr(avatar_session, 'quit_event', None) is None:
        return
    avatar_session.flush_talk()
    avatar_session.quit_event.set()

class SessionManager:
    """
    全局数字人会话管理器。
    
    统一管理 avatar_sessions 生命周期，并在脱离 WebRTC 时依然保持服务可用。
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.sessions: Dict[str, BaseAvatar] = {}
            self.build_session_fn = None
            # Sessions killed while their (slow) build was still running in the
            # executor. The slot is a `None` placeholder then, so there is no
            # quit_event to set yet — the builder publishes the session instead
            # of the caller, so it has to learn about the kill on return.
            self.killed: set = set()
            self.max_session = 1   # default, override via set_max_session()
            self.initialized = True

    def set_max_session(self, n: int):
        """设置最大并发会话数"""
        self.max_session = max(1, n)

    def init_builder(self, build_session_fn):
        """配置用于构建 avatar_session 的工厂函数"""
        self.build_session_fn = build_session_fn
        
    def get_session(self, sessionid: str) -> Optional[BaseAvatar]:
        """获取已存活的会话"""
        return self.sessions.get(sessionid)

    def has_session(self, sessionid: str) -> bool:
        """检查会话是否存在"""
        return sessionid in self.sessions and self.sessions[sessionid] is not None

    def active_sessions(self) -> int:
        """当前活跃（非占位）会话数"""
        return sum(1 for s in self.sessions.values() if s is not None)
        
    async def create_session(self, params: dict, sessionid: str = None) -> str:
        """
        在异步环境中创建一个新会话
        如果 sessionid 为 None，则自动生成。
        """
        if self.build_session_fn is None:
            raise Exception("SessionManager builder not initialized")
            
        if sessionid is None:
            sessionid = _rand_session_id()
            
        # 检查是否达到最大会话数
        active_count = sum(1 for s in self.sessions.values() if s is not None)
        if active_count >= self.max_session:
            raise MaxSessionError(
                f"Maximum session limit reached ({active_count}/{self.max_session})"
            )

        logger.info('Creating sessionid=%s, current session num=%d', sessionid, active_count)
        # 预先占位防止重复
        self.killed.discard(sessionid)
        self.sessions[sessionid] = None

        # 在线程池中构建 session（加载模型非常耗时）
        try:
            avatar_session = await asyncio.get_event_loop().run_in_executor(
                None, self.build_session_fn, sessionid, params
            )
        except BaseException:
            # A failed build (model load error, worker unreachable) used to
            # leave the None placeholder behind forever: invisible in the
            # admin console (it lists only real sessions) and never cleaned
            # up by anything. Drop it so a retry starts from a clean slate.
            self.killed.discard(sessionid)
            if self.sessions.get(sessionid) is None:
                self.sessions.pop(sessionid, None)
            raise
        if sessionid in self.killed:
            # 构建期间被终止：占位已被移除，直接丢弃刚建好的会话
            self.killed.discard(sessionid)
            logger.info('Session %s was killed during build — discarding', sessionid)
            _release(avatar_session)
            raise SessionKilledError(f"session {sessionid} was terminated during creation")
        self.sessions[sessionid] = avatar_session
        return sessionid
        
    def add_session(self, sessionid: str, avatar_session: BaseAvatar):
        """同步添加静态或外部管理的会话（供非服务端入口调用）"""
        self.sessions[sessionid] = avatar_session
        
    def kill_session(self, sessionid: str) -> bool:
        """终止单个会话，返回是否真的找到了它。

        构建中（占位为 None）的会话没有 quit_event 可置位，记入 killed，
        由 create_session 在构建返回时丢弃结果；建好后再销毁会泄漏
        render/inference 线程和 GPU 显存。
        """
        if sessionid not in self.sessions:
            return False
        avatar_session = self.sessions.get(sessionid)
        if avatar_session is None:
            logger.info("Killing session %s while it is still building", sessionid)
            self.killed.add(sessionid)
            self.sessions.pop(sessionid, None)
            return True
        logger.info("Killing session %s", sessionid)
        self.sessions.pop(sessionid, None)
        try:
            _release(avatar_session)
        except Exception:
            logger.exception(f"session {sessionid} cleanup error")
        return True

    def kill_all_sessions(self) -> list:
        """终止所有会话，返回被终止的 sessionid 列表。"""
        killed = []
        for sid in list(self.sessions.keys()):
            if self.sessions.get(sid) is not None:
                killed.append(sid)
            self.kill_session(sid)
        return killed

    def remove_session(self, sessionid: str):
        """销毁会话资源：置位 quit_event，级联停掉 render/推理/合帧/TTS 线程"""
        if sessionid in self.sessions:
            logger.info(f"Removing session {sessionid}")
            avatar_session = self.sessions.pop(sessionid, None)
            try:
                _release(avatar_session)
            except Exception:
                logger.exception(f"session {sessionid} cleanup error")

# 单例抛出
session_manager = SessionManager()
