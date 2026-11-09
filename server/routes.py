###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
###############################################################################

import json
import asyncio
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
# Lazy imports from app: the app module is fully loaded by the time these
# handlers run, so importing inside the function avoids the circular import.

async def model_status(request):
    """模型加载状态（lazy-loaded on first connect）"""
    try:
        import app as app_module
        opt = request.app.get("opt")
        return json_ok(data={
            "state": app_module.model_state,
            "model": getattr(opt, "model", ""),
            "active_sessions": session_manager.active_sessions(),
        })
    except Exception as e:
        logger.exception('model_status exception:')
        return json_error(str(e))


async def model_unload(request):
    """手动卸载模型（断开所有会话后仍保留, 需 force=true 才强制断开）"""
    try:
        import app as app_module
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
        import app as app_module
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


# ─── Remote GPU worker 配置 / 连通性测试 ────────────────────────────────────
# Settings history: data/remote_worker.json (admin UI) > CLI/env. The test
# endpoint performs a real gRPC Health call, so a green result means the port
# is reachable AND a worker is actually serving on it (not just an open socket).

def _probe_worker_blocking(address: str, timeout: float) -> dict:
    """Blocking connect + Health + list of what is resident. Runs in executor."""
    from remote.client import RemoteClient
    client = RemoteClient(address, timeout=timeout)
    try:
        info = client.health()
        return {"reachable": True, "health": info}
    finally:
        client.close()


def _friendly_probe_error(exc: Exception) -> str:
    """Turn a gRPC RpcError into something readable in a browser alert.

    The raw exception carries a multi-line debug string and a trailing '>' that
    is noise in a UI. Keep the useful part: transport vs. protocol failure.
    """
    text = str(exc)
    if "Connection refused" in text or "failed to connect" in text:
        return "connection refused — nothing is listening on that host:port"
    if "StatusCode.UNAVAILABLE" in text:
        return "worker unavailable (host unreachable, wrong port, or firewall)"
    if "StatusCode.DEADLINE_EXCEEDED" in text or "Deadline Exceeded" in text:
        return "timed out — the host responded too slowly"
    if "StatusCode.UNIMPLEMENTED" in text:
        return ("something is listening but it is not a LiveTalking worker "
                "(wrong port?)")
    # Fall back to the first line only.
    first = text.strip().splitlines()[0] if text.strip() else type(exc).__name__
    return first[:300]


async def remote_worker_status(request):
    """GET /api/remote/status — effective address, its source, CLI/env fallback."""
    try:
        from remote import settings
        addr, source = settings.effective()
        return json_ok(data={
            "address": addr,
            "source": source,
            "saved": settings.saved_address(),
            "env": settings.env_address(),
            "credentials": settings.credentials_summary(),
        })
    except Exception as e:
        logger.exception('remote_worker_status exception:')
        return json_error(str(e))


async def remote_credentials_save(request):
    """POST /api/remote/credentials {"username","host","password"}.

    Password goes to the OS keychain; username/host to the settings file.
    The response NEVER echoes the password back.
    """
    try:
        from remote import settings
        params = {}
        if request.can_read_body:
            try:
                params = await request.json()
            except Exception:
                params = {}
        settings.save_credentials(
            params.get("username", ""),
            params.get("host", ""),
            params.get("password", ""))
        return json_ok(data={"credentials": settings.credentials_summary()})
    except ValueError as e:
        return json_error(str(e))
    except RuntimeError as e:
        # keychain unavailable / package missing
        return json_error(str(e))
    except Exception as e:
        logger.exception('remote_credentials_save exception:')
        return json_error(str(e))


async def remote_credentials_clear(request):
    """POST /api/remote/credentials/clear — delete the keychain entry + metadata."""
    try:
        from remote import settings
        settings.clear_credentials()
        return json_ok(data={"credentials": settings.credentials_summary()})
    except Exception as e:
        logger.exception('remote_credentials_clear exception:')
        return json_error(str(e))


def _ssh_probe_blocking(username: str, host: str, password: str, port: int,
                        expected_fp, trust_new: bool) -> dict:
    from remote import ssh_test
    return ssh_test.test_ssh(username, host, password, port=port,
                             expected_fingerprint=expected_fp,
                             trust_new=trust_new)


async def remote_ssh_test(request):
    """POST /api/remote/ssh-test — real SSH login using the stored credentials.

    Body may override host/port. Body password is optional: when omitted, the
    stored keychain password is used (so the UI never has to hold it).
    """
    try:
        import asyncio
        from remote import settings, ssh_test

        params = {}
        if request.can_read_body:
            try:
                params = await request.json()
            except Exception:
                params = {}

        stored = settings.get_credentials() or {}
        username = (params.get("username") or stored.get("username") or "").strip()
        host = (params.get("host") or stored.get("host") or "").strip()
        password = params.get("password") or stored.get("password") or ""
        port = int(params.get("port") or ssh_test.DEFAULT_SSH_PORT)
        trust_new = bool(params.get("trust_new"))

        if not username or not host:
            return json_error("username and host are required")
        if not password:
            return json_error("no password — save credentials first, or supply one")

        expected_fp = settings.pinned_host_keys().get(host)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _ssh_probe_blocking, username, host, password, port,
            expected_fp, trust_new)

        # Pin on success so a later key change is refused.
        if result.get("ok") and result.get("fingerprint"):
            settings.pin_host_key(host, result["fingerprint"])

        return json_ok(data=result)
    except ValueError as e:
        return json_error(str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception('remote_ssh_test exception:')
        return json_ok(data={"ok": False, "message": str(e)})


async def remote_worker_save(request):
    """POST /api/remote/save {"address": "host:port"|""} — persist the address."""
    try:
        from remote import settings
        params = {}
        if request.can_read_body:
            try:
                params = await request.json()
            except Exception:
                params = {}
        address = params.get("address", "")
        if not (address or "").strip():
            settings.clear_saved()
        else:
            settings.save_address(address)
        addr, source = settings.effective()
        return json_ok(data={"address": addr, "source": source})
    except ValueError as e:
        return json_error(str(e))
    except Exception as e:
        logger.exception('remote_worker_save exception:')
        return json_error(str(e))


async def remote_worker_test(request):
    """POST /api/remote/test {"address": "host:port"}? — real gRPC Health probe.

    The address in the body is tested WITHOUT saving it, so you can check a
    host before committing it. Falls back to the effective address when absent
    or empty.
    """
    address = ""
    try:
        from remote import settings
        params = {}
        if request.can_read_body:
            try:
                params = await request.json()
            except Exception:
                params = {}
        raw = params.get("address")
        address = settings.normalize(raw) if (raw or "").strip() else settings.effective_address()
        if not address:
            return json_error("no address to test — enter one first")
        # gRPC is blocking; keep the event loop free.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _probe_worker_blocking, address, 5.0)
        info = result.get("health", {})
        return json_ok(data={
            "address": address,
            "reachable": True,
            "device": info.get("device", ""),
            "gpu": info.get("detail", ""),
            "backend": info.get("backend", ""),
            "avatar_id": info.get("avatar_id", ""),
            "cuda": info.get("device") == "cuda",
        })
    except ValueError as e:
        return json_error(str(e))
    except Exception as e:  # noqa: BLE001
        # Unreachable is an expected outcome, not a server error: report it as a
        # normal (code 0) payload with reachable=false so the UI can render it.
        logger.warning('[remote] test failed for %s: %s', address or "?", e)
        return json_ok(data={"address": address, "reachable": False,
                             "error": _friendly_probe_error(e)})


async def admin_sessions(request):
    """Admin: 获取活跃的会话及其配置"""
    try:
        sessions_info = []
        for sid, avatar_session in session_manager.sessions.items():
            if avatar_session:
                s_opt = getattr(avatar_session, 'opt', None)
                s_data = {
                    "sessionid": sid,
                    "speaking": avatar_session.is_speaking() if hasattr(avatar_session, 'is_speaking') else False,
                    "recording": getattr(avatar_session, 'recording', False),
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
    app.router.add_get("/api/remote/status", remote_worker_status)
    app.router.add_post("/api/remote/save", remote_worker_save)
    app.router.add_post("/api/remote/test", remote_worker_test)
    app.router.add_post("/api/remote/credentials", remote_credentials_save)
    app.router.add_post("/api/remote/credentials/clear", remote_credentials_clear)
    app.router.add_post("/api/remote/ssh-test", remote_ssh_test)
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
