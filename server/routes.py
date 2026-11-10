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
    """GET /api/remote/status — effective address, its source, compute mode."""
    try:
        from remote import revision, settings, tunnel
        addr, source = settings.effective()
        summary = {
            "mode": settings.worker_mode(),
            "remote_active": settings.remote_active(),
            "address": addr,
            "source": source,
            "saved": settings.saved_address(),
            "env": settings.env_address(),
            "credentials": settings.credentials_summary(),
            # Drift signal for the card: the host's own code revision plus
            # whether the model defaults to the remote worker.
            "host_revision": revision.current_revision(),
            "host_model": getattr(request.app.get("opt"), "model", ""),
            # SSH-only transport: the worker binds loopback on the box and the
            # host reaches it through `ssh -L`. This block tells the card what
            # the tunnel is doing right now.
            "tunnel": {**tunnel.status(),
                       "ssh_port": settings.ssh_port(),
                       "worker_port": settings.worker_port()},
        }
        return json_ok(data=summary)
    except Exception as e:
        logger.exception('remote_worker_status exception:')
        return json_error(str(e))


async def remote_mode(request):
    """GET/POST /api/remote/mode — the local/remote compute toggle.

    POST body {"mode": "local"|"remote"}. The value is persisted in
    data/remote_worker.json and read per dispatch, so it applies to the next
    session / avatar task / enhance job without restarting the server. Switching
    while a model is loaded is allowed but the caller should unload first: the
    resident weights belong to the previous mode.
    """
    try:
        from remote import settings
        if request.method == "GET":
            return json_ok(data={"mode": settings.worker_mode(),
                                 "remote_active": settings.remote_active()})
        params = {}
        if request.can_read_body:
            try:
                params = await request.json()
            except Exception:
                params = {}
        mode = settings.set_worker_mode(params.get("mode", ""))
        # Keep opt.model in step with the toggle so /api/admin/config and the
        # next connect agree on where work runs.
        app_module = _app_module()
        opt = request.app.get("opt")
        if opt is not None:
            if mode == settings.MODE_REMOTE:
                opt.model = "remote"
            elif opt.model == "remote":
                opt.model = getattr(opt, "remote_backend", "") or "musetalk"
        if mode == settings.MODE_REMOTE:
            # Bring the SSH forward up and probe the worker in the background:
            # the toggle must not block on an SSH handshake, but the next
            # connect should find the tunnel open and the supervisor running.
            from remote import tunnel
            if settings.tunnel_enabled():
                tunnel.resume()
                # start_background_supervisor is idempotent and ensures once
                # synchronously; run it off the event loop so the response is
                # not stuck behind an SSH handshake.
                asyncio.get_running_loop().run_in_executor(
                    None, tunnel.start_background_supervisor)
            announce = getattr(app_module, "_announce_remote", None)
            if announce is not None:
                threading.Thread(target=announce, args=(opt,),
                                 name="remote-announce", daemon=True).start()
        resident = getattr(app_module, "model_backend", "")
        return json_ok(data={
            "mode": mode,
            "remote_active": settings.remote_active(),
            "address": settings.effective_address(),
            "resident_backend": resident,
            # Switching modes does not free the loaded weights; say so instead
            # of letting the next connect fail with a backend-swap refusal.
            "reload_required": bool(resident) and (
                (mode == settings.MODE_REMOTE) != (resident == "remote")),
        })
    except ValueError as e:
        return json_error(str(e))
    except Exception as e:
        logger.exception('remote_mode exception:')
        return json_error(str(e))


def _connect_steps_blocking(username: str, host: str, password: str, port: int,
                            trust_new: bool, login: dict | None = None) -> dict:
    """Everything "Connect" does, off the event loop.

    One button, four things that can each fail for a different reason: log in
    with the password, authorize this host's key (the forward cannot prompt),
    prove a key-only login, and open the tunnel. Each outcome is reported
    separately so the UI can point at the failing step instead of saying
    "connection failed".

    ``login`` lets the caller run the password check first and only persist the
    credentials when it succeeded -- a typo'd host must not overwrite a working
    configuration.
    """
    from remote import keyinstall, settings, ssh_test, tunnel

    steps = []

    def step(name, ok, detail=""):
        steps.append({"name": name, "ok": bool(ok), "detail": detail})
        return ok

    if login is None:
        expected_fp = settings.pinned_host_keys().get(host)
        login = ssh_test.test_ssh(username, host, password, port=port,
                                  expected_fingerprint=expected_fp,
                                  trust_new=trust_new or not expected_fp)
    detail = login.get("message", "")
    if login.get("ok") and login.get("fingerprint"):
        detail += " · host key %s..." % login["fingerprint"][:16]
    if not step("ssh-login", login.get("ok"), detail):
        return {"ok": False, "steps": steps, "tunnel": tunnel.status()}
    if login.get("fingerprint"):
        settings.pin_host_key(host, login["fingerprint"])

    install = keyinstall.ensure_installed(
        username, host, password, port=port,
        expected_fingerprint=settings.pinned_host_keys().get(host))
    if not step("authorize-key", install.get("ok"), install.get("message", "")):
        return {"ok": False, "steps": steps, "tunnel": tunnel.status()}

    verify = keyinstall.verify_key_login(username, host, port=port)
    if not step("key-login", verify.get("ok"), verify.get("message", "")):
        return {"ok": False, "steps": steps, "tunnel": tunnel.status()}

    # The worker address is host:worker_port — the tunnel targets the same host,
    # so the saved identity is all the rest of the app needs.
    tunnel.resume()
    if settings.tunnel_enabled():
        opened = tunnel.ensure(timeout=20.0)
        if not step("tunnel", opened,
                    "127.0.0.1:%d forwarded" % tunnel.DEFAULT_LOCAL_PORT
                    if opened else "the forward did not come up; see data/tunnel.log"):
            return {"ok": False, "steps": steps, "tunnel": tunnel.status()}
    else:
        step("tunnel", True, "direct mode (LT_SSH_TUNNEL=0): no forward")

    address = settings.effective_address()
    health = {}
    error = ""
    try:
        from remote.client import probe
        health = probe(address, timeout=8.0)
    except Exception as exc:  # noqa: BLE001
        error = _friendly_probe_error(exc)
    step("worker", bool(health), error or "worker answered on %s" % address)

    from remote import revision as revision_mod
    remote_rev = health.get("revision", "")
    return {
        "ok": bool(health),
        "steps": steps,
        "address": address,
        "device": health.get("device", ""),
        "gpu": health.get("detail", ""),
        "busy": bool(health.get("busy")),
        "revision": remote_rev,
        "host_revision": revision_mod.current_revision(),
        "revision_match": revision_mod.matches(remote_rev,
                                               revision_mod.current_revision()),
        "error": error,
        "tunnel": tunnel.status(),
    }


async def remote_connect(request):
    """POST /api/remote/connect {username, host, password?, port?}.

    The one-shot path behind the console's Connect button: store the login,
    authorize this host's key on the box, open the SSH forward and probe the
    worker. Password is optional when the keychain already holds one.
    """
    try:
        from remote import settings

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
        port = int(params.get("port") or settings.ssh_port() or 22)

        if not username or not host:
            return json_error("host and username are required")
        if not password:
            return json_error("password is required the first time (it is stored in "
                              "the OS keychain afterwards)")

        expected_fp = settings.pinned_host_keys().get(host)
        loop = asyncio.get_event_loop()
        login = await loop.run_in_executor(
            None, _ssh_probe_blocking, username, host, password, port,
            expected_fp, not expected_fp)

        # Persist only after the login worked: a typo'd host must not replace a
        # working configuration (nor a good keychain password).
        if login.get("ok"):
            settings.save_credentials(username, host, password)
            settings.set_ssh_port(port)

        result = await loop.run_in_executor(
            None, _connect_steps_blocking, username, host, password, port,
            True, login)
        if "credentials" not in result:
            result["credentials"] = settings.credentials_summary()
        return json_ok(data=result)
    except ValueError as e:
        return json_error(str(e))
    except RuntimeError as e:
        return json_error(str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception('remote_connect exception:')
        return json_error(str(e))


async def remote_disconnect(request):
    """POST /api/remote/disconnect — close the SSH forward and keep it closed."""
    try:
        from remote import tunnel
        tunnel.suspend()
        return json_ok(data={"tunnel": tunnel.status()})
    except Exception as e:
        logger.exception('remote_disconnect exception:')
        return json_error(str(e))


async def remote_tunnel_control(request):
    """POST /api/remote/tunnel {"action": "start"|"stop"|"mode",
    "enabled": bool} — control the SSH forward from the admin console."""
    try:
        from remote import settings, tunnel
        params = {}
        if request.can_read_body:
            try:
                params = await request.json()
            except Exception:
                params = {}
        action = (params.get("action") or "status").strip().lower()
        if action == "mode":
            settings.set_tunnel(bool(params.get("enabled", True)))
            if settings.tunnel_enabled():
                tunnel.resume()
                loop = asyncio.get_event_loop()
                ok = await loop.run_in_executor(None, tunnel.ensure)
                return json_ok(data={"tunnel": tunnel.status(), "opened": ok})
            tunnel.stop()
        elif action == "start":
            tunnel.resume()
            loop = asyncio.get_event_loop()
            ok = await loop.run_in_executor(None, tunnel.ensure)
            if not ok:
                return json_error(
                    "could not open the tunnel — check SSH key access and "
                    "data/tunnel.log")
        elif action == "stop":
            tunnel.stop()
        return json_ok(data={"tunnel": tunnel.status()})
    except Exception as e:
        logger.exception('remote_tunnel_control exception:')
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
        from remote import revision as revision_mod
        remote_rev = info.get("revision", "")
        local_rev = revision_mod.current_revision()
        return json_ok(data={
            "address": address,
            "reachable": True,
            "device": info.get("device", ""),
            "gpu": info.get("detail", ""),
            "backend": info.get("backend", ""),
            "avatar_id": info.get("avatar_id", ""),
            "cuda": info.get("device") == "cuda",
            "busy": bool(info.get("busy")),
            # Version drift is the difference between "the box answers" and
            # "the box runs my code". Empty remote_revision = pre-revision build.
            "revision": remote_rev,
            "host_revision": local_rev,
            "revision_match": revision_mod.matches(remote_rev, local_rev),
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
    app.router.add_get("/api/remote/status", remote_worker_status)
    app.router.add_get("/api/remote/mode", remote_mode)
    app.router.add_post("/api/remote/mode", remote_mode)
    app.router.add_post("/api/remote/connect", remote_connect)
    app.router.add_post("/api/remote/disconnect", remote_disconnect)
    app.router.add_post("/api/remote/save", remote_worker_save)
    app.router.add_post("/api/remote/test", remote_worker_test)
    app.router.add_post("/api/remote/credentials", remote_credentials_save)
    app.router.add_post("/api/remote/credentials/clear", remote_credentials_clear)
    app.router.add_post("/api/remote/ssh-test", remote_ssh_test)
    app.router.add_post("/api/remote/tunnel", remote_tunnel_control)
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
