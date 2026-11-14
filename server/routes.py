###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
###############################################################################

import json
import asyncio
import threading
from aiohttp import web

from utils.logger import logger


# ─── 路由工具函数 ──────────────────────────────────────────────────────────

def json_ok(data=None):
    """返回成功 JSON 响应"""
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )


def json_error(msg: str, code: int = -1):
    """返回错误 JSON 响应"""
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


from server.session_manager import session_manager
from server.avatar_routes import setup_avatar_routes

def get_session(request, sessionid: str):
    """从 app 中获取 session 实例"""
    return session_manager.get_session(sessionid)


# ─── 路由处理函数 ──────────────────────────────────────────────────────────

async def human(request):
    """文本输入（echo/chat 模式），支持 voice/emotion 参数"""
    try:
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get('interrupt'):
            avatar_session.flush_talk()

        datainfo = {}
        if params.get('tts'):  # tts 参数透传（voice, emotion 等）
            datainfo['tts'] = params.get('tts')

        if params['type'] == 'echo':
            avatar_session.put_msg_txt(params['text'], datainfo)
        elif params['type'] == 'chat':
            llm_response = request.app.get("llm_response")
            if llm_response:
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params['text'], avatar_session, datainfo
                )

        return json_ok()
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """打断当前说话"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        return json_ok()
    except Exception as e:
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))


async def humanaudio(request):
    """上传音频文件"""
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.put_audio_file(filebytes, datainfo)
        return json_ok()
    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))


async def set_audiotype(request):
    """设置自定义状态（动作编排）"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """录制控制"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params['type'] == 'start_record':
            avatar_session.start_recording()
        elif params['type'] == 'end_record':
            avatar_session.stop_recording()
        return json_ok()
    except Exception as e:
        logger.exception('record exception:')
        return json_error(str(e))


async def is_speaking(request):
    """查询是否正在说话"""
    params = await request.json()
    sessionid = params.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())

async def sse_handler(request):
    """SSE 事件流，推送服务器状态更新到客户端"""
    sessionid = request.query.get('sessionid', '')
    avatar_session = session_manager.get_session(sessionid)
    if avatar_session is None:
        return json_error("session not found")

    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        }
    )
    await response.prepare(request)

    import queue
    msgqueue = queue.Queue()
    avatar_session.add_msgqueue(msgqueue)

    try:
        while True:
            try:
                msg = msgqueue.get_nowait()
                await response.write(f"data: {msg}\n\n".encode('utf-8'))
            except queue.Empty:
                await asyncio.sleep(0.01)
    except (asyncio.CancelledError, ConnectionResetError):
        logger.info('SSE connection closed for session: %s', sessionid)
    finally:
        if msgqueue in avatar_session.msgqueues:
            avatar_session.msgqueues.remove(msgqueue)

    return response


# ─── 模型生命周期 / 应用重启 ────────────────────────────────────────────────
# app.py is normally launched as __main__ (./run.sh → python app.py). A plain
# `import app` from here then creates a SECOND module object whose globals are
# untouched: model_state would always read 'unloaded' and unload_model() would
# free nothing, because it ran against the phantom copy. Resolve the running
# module instead.
def _app_module():
    import sys
    main = sys.modules.get('__main__')
    if main is not None and str(getattr(main, '__file__', '')).endswith('app.py'):
        return main
    import app as app_module
    return app_module


async def model_status(request):
    """模型加载状态（lazy-loaded on first connect）"""
    try:
        app_module = _app_module()
        opt = request.app.get("opt")
        return json_ok(data={
            "state": app_module.model_state,
            "backend": app_module.model_backend,
            "model": getattr(opt, "model", ""),
            "active_sessions": session_manager.active_sessions(),
        })
    except Exception as e:
        logger.exception('model_status exception:')
        return json_error(str(e))


async def model_unload(request):
    """手动卸载模型（断开所有会话后仍保留, 需 force=true 才强制断开）"""
    try:
        app_module = _app_module()
        params = {}
        if request.can_read_body:
            try:
                params = await request.json()
            except Exception:
                params = {}
        force = bool(params.get('force', False))
        msg = await asyncio.get_event_loop().run_in_executor(
            None, lambda: app_module.unload_model(force=force))
        return json_ok(data={"result": msg})
    except Exception as e:
        logger.exception('model_unload exception:')
        return json_error(str(e))


async def app_restart(request):
    """重启整个应用（通过 restart.sh, 保留原启动参数）"""
    try:
        app_module = _app_module()
        app_module.restart_app()
        return json_ok(data={"result": "restarting — the server will be back in a few seconds"})
    except Exception as e:
        logger.exception('app_restart exception:')
        return json_error(str(e))


async def admin_config(request):
    """Admin: 获取全局配置参数"""
    try:
        opt = request.app.get("opt")
        if opt:
            return json_ok(data={"config": vars(opt)})
        return json_error("Config not found")
    except Exception as e:
        logger.exception('admin_config exception:')
        return json_error(str(e))


async def admin_sessions(request):
    """Admin: 获取活跃的会话及其配置

    包含仍在构建中的会话 (占位为 None)：首次连接要加载模型，可能长达数十秒，
    这期间它占用一个 max_session 名额却不能从控制台看到 —— 正是最需要
    管理员能看见并终止的时刻。
    """
    try:
        sessions_info = []
        for sid, avatar_session in session_manager.sessions.items():
            if avatar_session is None:
                # 构建中：没有 opt/tracks 可读，只报告它是谁
                sessions_info.append({
                    "sessionid": sid,
                    "speaking": False,
                    "recording": False,
                    "building": True,
                })
                continue
            s_opt = getattr(avatar_session, 'opt', None)
            s_data = {
                "sessionid": sid,
                "speaking": avatar_session.is_speaking() if hasattr(avatar_session, 'is_speaking') else False,
                "recording": getattr(avatar_session, 'recording', False),
                "building": False,
            }
            if s_opt:
                s_data.update({
                    "model": getattr(s_opt, "model", ""),
                    "avatar_id": getattr(s_opt, "avatar_id", ""),
                    "REF_FILE": getattr(s_opt, "REF_FILE", ""),
                    "transport": getattr(s_opt, "transport", ""),
                    "batch_size": getattr(s_opt, "batch_size", 0),
                    "customopt": getattr(s_opt, "customopt", []),
                })
            sessions_info.append(s_data)
        return json_ok(data={"sessions": sessions_info})
    except Exception as e:
        logger.exception('admin_sessions exception:')
        return json_error(str(e))


# ─── 会话终止 ──────────────────────────────────────────────────────────────
# Two halves, both required: the avatarsession owns the render/inference
# threads and the PeerConnection owns the browser's live transceiver. Killing
# only the former leaves a "connected" player that never receives frames;
# killing only the latter leaves the threads and the GPU memory resident.

def _rtc_manager(request):
    return request.app.get("rtc_manager")


async def _stop_session(manager, sessionid: str) -> bool:
    """Stop the avatar FIRST, then close its peer connection.

    Order is not cosmetic: closing the pc flips its connection state to
    "closed", and RTCManager's ``connectionstatechange`` handler calls
    ``remove_session`` — which pops the session and reports success for a
    session we no longer know about. Killing the session first makes the
    answer ("killed" vs "not found") describe what actually happened, and
    the later remove_session simply finds nothing to do.
    """
    found = await asyncio.get_event_loop().run_in_executor(
        None, session_manager.kill_session, sessionid)
    if manager is not None:
        await manager.close_session(sessionid)
    return found


async def admin_session_kill(request):
    """Admin: 终止单个会话（断开其浏览器连接并停掉渲染线程）"""
    try:
        sessionid = request.match_info.get("sessionid", "")
        if not sessionid:
            return json_error("sessionid is required")
        found = await _stop_session(_rtc_manager(request), sessionid)
        if not found:
            return json_error(f"session {sessionid} not found")
        return json_ok(data={"sessionid": sessionid, "killed": True,
                             "active_sessions": session_manager.active_sessions()})
    except Exception as e:
        logger.exception('admin_session_kill exception:')
        return json_error(str(e))


async def admin_sessions_kill_all(request):
    """Admin: 终止所有会话"""
    try:
        manager = _rtc_manager(request)
        # Stop the sessions first so the returned list is the truth: closing
        # the pcs first would let connectionstatechange → remove_session empty
        # the registry, and this would answer "killed: []" after killing N.
        killed = await asyncio.get_event_loop().run_in_executor(
            None, session_manager.kill_all_sessions)
        if manager is not None:
            for sid in killed:
                await manager.close_session(sid)
        return json_ok(data={"killed": killed, "count": len(killed),
                             "active_sessions": session_manager.active_sessions()})
    except Exception as e:
        logger.exception('admin_sessions_kill_all exception:')
        return json_error(str(e))


# ─── 路由注册 ──────────────────────────────────────────────────────────────

async def index(request):
    """默认首页重定向"""
    opt = request.app.get("opt")
    pagename = 'index.html'
    if opt and opt.transport == 'rtmp':
        pagename = 'rtmpapi.html'
    elif opt and opt.transport == 'rtcpush':
        pagename = 'rtcpushapi.html'
    raise web.HTTPFound(f'/{pagename}')


def setup_routes(app):
    """注册所有路由到 aiohttp app"""
    app.router.add_get("/", index)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_get("/api/admin/config", admin_config)
    app.router.add_get("/api/admin/sessions", admin_sessions)
    app.router.add_post("/api/admin/sessions/kill-all", admin_sessions_kill_all)
    app.router.add_post("/api/admin/sessions/{sessionid}/kill", admin_session_kill)
    app.router.add_get("/api/model/status", model_status)
    app.router.add_post("/api/model/unload", model_unload)
    app.router.add_post("/api/app/restart", app_restart)
    app.router.add_get('/sse', sse_handler)

    # ── Local ASR endpoint (SenseVoice/FunASR) ── Issue #604 ──
    try:
        from server.asr_server import asr_websocket_handler, is_funasr_available
        if is_funasr_available():
            app.router.add_get("/api/asr", asr_websocket_handler)
            logger.info("[ASR] Local SenseVoice ASR endpoint enabled at /api/asr")
        else:
            logger.info("[ASR] funasr not installed — local ASR endpoint disabled "
                        "(pip install funasr modelscope)")
    except Exception as e:
        logger.warning(f"[ASR] Failed to register ASR endpoint: {e}")

    # 注册 avatar 生成相关的路由
    setup_avatar_routes(app)

    # 注册视频增强（base video 预处理）路由
    try:
        from server.enhance_routes import setup_enhance_routes
        setup_enhance_routes(app)
        logger.info("[Enhance] /api/enhance/* routes enabled")
    except Exception as e:
        logger.warning(f"[Enhance] failed to register enhance routes: {e}")

    app.router.add_static('/', path='web')
