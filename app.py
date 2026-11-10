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

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
#from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import os
import numpy as np
from threading import Thread,Event
import threading
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
import utils.aioice_fix  # noqa: F401  — must load before aiortc creates ICE connections
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from server.webrtc import HumanPlayer
from avatars.base_avatar import BaseAvatar
from llm import llm_response
import registry
from server.routes import setup_routes
from server.rtc_manager import RTCManager
from server.session_manager import session_manager

import argparse
import random
import shutil
import signal
import atexit
import asyncio
import torch
from io import BytesIO
from typing import Dict
from utils.logger import logger
import copy
import gc
from dotenv import load_dotenv


app = Flask(__name__)
#sockets = Sockets(app)
opt = None
model = None
global_avatars = {} # avatar_id: payload

# Fix 4: set once the avatar model has finished loading + warming up, so the
# HTTP layer can bind immediately and session creation blocks cleanly until
# the model is ready.
model_ready_event = threading.Event()

# ─── Lazy model loading ──────────────────────────────────────────────────────
# Weights are NOT loaded at startup: the first connect triggers
# ensure_model_loaded(). After that the model stays in memory across
# disconnects; unload_model() frees it manually.
model_state = 'unloaded'  # 'unloaded' | 'loading' | 'ready'
model_state_lock = threading.Lock()
# Serialises the whole connect sequence (resolve backend -> load weights ->
# build session). Without it, two simultaneous connects for DIFFERENT backends
# could swap the weights between one caller's ensure_model_loaded() and its
# session construction, leaving that session pointing at freed weights.
connect_lock = threading.RLock()

# PID file: restart.sh uses it to stop the exact process that owns the port.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PID_FILE = os.path.join(BASE_DIR, 'livetalking.pid')


def _write_pidfile():
    try:
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
    except OSError:
        logger.warning('[startup] could not write pid file %s', PID_FILE)


def _remove_pidfile():
    """Remove the pid file only if it still belongs to this process."""
    try:
        with open(PID_FILE) as f:
            owner = f.read().strip()
        if owner == str(os.getpid()):
            os.remove(PID_FILE)
    except (OSError, ValueError):
        pass
        

#####lazy model management#################################
# Backend module per avatar model name. Hoisted out of main() so the connect
# path can (re)bind to any backend, not only the one --model selected at start.
_AVATAR_MODULES = {
    'musetalk':   'avatars.musetalk_avatar',
    'wav2lip':    'avatars.wav2lip_avatar',
    'ultralight': 'avatars.ultralight_avatar',
    'remote':     'avatars.remote_avatar',
}

# Backends used by the remote WORKER (ultralight is host-only).
_REMOTE_WORKER_BACKENDS = ('musetalk', 'wav2lip')

# Which backend the in-memory `model` currently holds. '' until the first load.
# For model=remote this is the HOST plugin name ('remote'); the worker-side
# backend is tracked inside avatars/remote_avatar.py.
model_backend = ''


def _bind_backend(backend: str):
    """Import `backend`'s module and return (load_model, load_avatar, warm_up)."""
    import importlib
    mod = importlib.import_module(_AVATAR_MODULES[backend])
    return mod.load_model, mod.load_avatar, mod.warm_up


def _warm_up_for(backend, model_obj, avatar_payload):
    """Call the backend's warm_up with the argument shape it expects.

    The four backends disagree: musetalk/remote take (batch, model), wav2lip
    takes (batch, model, resolution), ultralight takes (batch, avatar, res).
    Keeping the table here means adding a backend is one line, not another
    elif-chain in two places.
    """
    lm, la, warm_up = _bind_backend(backend)
    if backend == 'wav2lip':
        warm_up(opt.batch_size, model_obj, 256)
    elif backend == 'ultralight':
        warm_up(opt.batch_size, avatar_payload, 160)
    else:  # musetalk, remote
        warm_up(opt.batch_size, model_obj)


def worker_remote() -> bool:
    """True when GPU work should be dispatched to the remote worker.

    This is the single decision point for the local/remote toggle: the value
    saved in the admin console (or --worker_mode / LT_WORKER_MODE) decides it,
    per call, so flipping the toggle takes effect on the next connect without a
    restart. An address is NOT required to answer "remote" here -- an
    unconfigured worker must fail loudly on connect, not silently run a
    multi-hour job on this machine.
    """
    from remote import settings
    return settings.worker_mode() == settings.MODE_REMOTE


def resolve_avatar_backend(avatar_id: str) -> str:
    """Pick the model that can actually drive `avatar_id`.

    The configured --model is a default, not a hard constraint: an avatar's
    artifacts dictate which backend can load it (latents.pt+mask/ => musetalk,
    face_imgs/ => wav2lip, ultralight.pth => ultralight). Trusting --model
    blindly is what made connecting a wav2lip avatar to a musetalk server die on
    a missing latents.pt.

    In remote mode the HOST plugin is 'remote' and the WORKER switches backend,
    so the artifact must map to a worker-supported backend. An avatar only this
    machine can drive (ultralight) still runs locally instead of failing the
    whole toggle.
    """
    from avatars import detect as avatar_detect
    detected, reason = avatar_detect.detect_backend(avatar_id)
    if detected is None:
        raise RuntimeError(f"cannot load avatar '{avatar_id}': {reason}")
    if not worker_remote():
        if detected != opt.model:
            logger.info('[model] avatar %r belongs to backend %r, not the '
                        'configured %r -- switching (%s)',
                        avatar_id, detected, opt.model, reason)
        return detected
    # Remote mode: the plugin is always 'remote'; the worker needs a backend it
    # can run.
    if detected not in _REMOTE_WORKER_BACKENDS:
        logger.info('[model] avatar %r needs backend %r, which the worker cannot '
                    'drive -- loading it locally (%s)', avatar_id, detected, reason)
        return detected
    return 'remote'


def _load_avatar_payload(backend: str, avatar_id: str):
    """load_avatar() for `backend`, but bind the backend's own loader first."""
    _lm, load_avatar, _wu = _bind_backend(backend)
    return load_avatar(avatar_id)


def _load_backend_weights(backend: str):
    """Load (or reload) the weights for the backend that should serve now."""
    global model, model_backend
    if model is not None and model_backend and model_backend != backend:
        # Drop the old weights BEFORE loading the new ones so the process never
        # holds two models at once (a musetalk+unet pair plus a wav2lip net is
        # more VRAM than the box has).
        model = None
        global_avatars.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    load_model, _load_avatar, _warm_up = _bind_backend(backend)
    # The four load_model() signatures disagree: musetalk takes nothing, wav2lip
    # wants the checkpoint path, ultralight/remote take opt.
    if backend == 'musetalk':
        model = load_model()
    elif backend == 'wav2lip':
        model = load_model("./models/wav2lip.pth")
    else:  # ultralight, remote
        model = load_model(opt)
    model_backend = backend


def default_backend() -> str:
    """The backend to load when the caller did not name one.

    --model is a default, not a switch: with `model: remote` in config.yaml a
    server toggled back to local must load a real local backend (and vice
    versa), otherwise the toggle would still create RemoteAvatar sessions.
    """
    configured = (opt.model if opt else '') or 'wav2lip'
    if worker_remote():
        return 'remote'
    if configured == 'remote':
        # 'remote' names no local backend; _REMOTE_WORKER_BACKENDS[0]-style
        # guessing is wrong (the avatar's artifacts decide later anyway), so
        # start from the configured worker backend, which is a real one.
        return getattr(opt, 'remote_backend', '') or 'musetalk'
    return configured


def _do_load_models(backend: str | None = None, avatar_id: str | None = None):
    """Load the weights for `backend` and cache `avatar_id`'s payload.

    `avatar_id` defaults to the configured --avatar_id, but the connect path
    passes the avatar it is actually about to drive so the warm-up uses the
    right payload (a musetalk payload is a 5-tuple, wav2lip a 3-tuple).
    """
    global model
    backend = backend or default_backend()
    assert backend, "no backend resolved for model load"
    avatar_id = avatar_id or opt.avatar_id
    _load_backend_weights(backend)
    global_avatars[avatar_id] = _load_avatar_payload(backend, avatar_id)
    _warm_up_for(backend, model, global_avatars[avatar_id])


def ensure_model_loaded(backend: str | None = None, avatar_id: str | None = None):
    """Lazy-load (or swap) avatar weights for `backend`. Thread-safe.

    `backend` defaults to default_backend(). When a connect targets an avatar
    whose artifacts need a different backend, the caller passes it and the
    weights are swapped (a single model is resident at a time by design).
    `avatar_id` is the avatar to load and warm up for that backend.

    The lock guards state transitions only -- never the load itself. Holding it
    across a multi-second (or multi-minute, remote) load would make every other
    connect block on the mutex instead of on model_ready_event, and made the
    original "someone else is loading" branch unreachable.
    """
    global model_state, model_backend
    backend = backend or default_backend()
    avatar_id = avatar_id or opt.avatar_id
    while True:
        with model_state_lock:
            if model_state == 'ready' and model_backend == backend:
                return
            if model_state == 'loading':
                # Another thread is loading; wait for it, then re-check whether
                # it loaded the backend THIS caller needs.
                waiter = True
            else:
                model_state = 'loading'
                model_ready_event.clear()
                waiter = False

        if waiter:
            model_ready_event.wait()
            if model_state != 'ready':
                raise RuntimeError('avatar model failed to load — check server logs')
            continue

        # This thread owns the load.
        try:
            if model_backend and model_backend != backend:
                # Swapping weights invalidates the resident model: a live session
                # of the old backend would keep calling inference_batch against
                # weights that no longer exist. Same stance as unload_model.
                active = session_manager.active_sessions()
                if active:
                    raise RuntimeError(
                        f"cannot switch the avatar backend from {model_backend!r} "
                        f"to {backend!r}: {active} session(s) are still connected. "
                        f"Disconnect them first, then connect the "
                        f"{backend!r} avatar.")
                logger.info('[model] switching backend %s -> %s', model_backend, backend)
            _do_load_models(backend, avatar_id)
            with model_state_lock:
                model_state = 'ready'
            logger.info('[model] backend %r loaded and warmed up (lazy load)', backend)
            return
        except Exception:
            # The load may have dropped the previous weights before failing, so
            # nothing coherent is resident: record that, don't leave a stale
            # model_backend claiming weights this process no longer holds.
            with model_state_lock:
                model_state = 'unloaded'
                model_backend = ''
            logger.exception('[model] model load/warmup failed')
            raise
        finally:
            model_ready_event.set()


def unload_model(force: bool = False) -> str:
    """Free model weights + cached avatars. Refuses while sessions are active
    unless force=True, which disconnects them first."""
    global model, model_state, model_backend
    # Take the same lock the connect path holds, so an unload cannot free the
    # weights underneath a session that is being built right now.
    with connect_lock:
        return _unload_model_locked(force)


def _unload_model_locked(force: bool) -> str:
    global model, model_state, model_backend
    with model_state_lock:
        for sid in list(session_manager.sessions.keys()):
            if session_manager.sessions[sid] is not None and not force:
                raise RuntimeError(
                    f'session {sid} is still connected — disconnect it first '
                    f'or call unload with force=true')
            session_manager.remove_session(sid)
        if model_state == 'unloaded':
            return 'model already unloaded'
        model = None
        global_avatars.clear()
        if model_backend == 'remote':
            try:
                from avatars.remote_avatar import unload_remote
                unload_remote()
            except Exception:  # noqa: BLE001
                logger.warning('[model] remote worker unload failed', exc_info=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_state = 'unloaded'
        model_backend = ''
        model_ready_event.clear()
        logger.info('[model] avatar model unloaded (weights freed)')
        return 'model unloaded'


def restart_app():
    """Relaunch the whole process via restart.sh with the same CLI args.

    The delayed spawn lets the HTTP response reach the client first; restart.sh
    then SIGTERMs this process, which triggers the graceful shutdown path.
    """
    import subprocess
    import sys
    import time
    cmd = ['/bin/bash', os.path.join(BASE_DIR, 'restart.sh')] + sys.argv[1:]
    def _spawn():
        time.sleep(0.7)
        logger.info('[restart] relaunching app: %s', ' '.join(cmd))
        subprocess.Popen(cmd, cwd=BASE_DIR, start_new_session=True)
    threading.Thread(target=_spawn, daemon=True).start()


#####webrtc###############################
# rtc_manager replaces the old pcs set and duplicate offer handlers.
rtc_manager = None

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_avatar_session(sessionid:str, params:dict)->BaseAvatar:
    # Lazy load: first connect triggers the (slow) weight load + warmup; later
    # connects reuse the in-memory model. Runs in a thread pool (via
    # SessionManager.create_session), so this blocks cleanly while the HTTP
    # dashboard stays responsive.
    #
    # The whole sequence is serialised: resolving a backend may swap the
    # resident weights, and a session built against weights that another
    # connect freed a moment later would crash mid-inference.
    with connect_lock:
        return _build_avatar_session_locked(sessionid, params)


def _build_avatar_session_locked(sessionid:str, params:dict)->BaseAvatar:
    opt_this = copy.deepcopy(opt)
    opt_this.sessionid = sessionid

    avatar_id = params.get('avatar', opt.avatar_id) or opt.avatar_id
    opt_this.avatar_id = avatar_id
    ref_audio = params.get('refaudio','') #音色
    ref_text = params.get('reftext','')

    # Pick the model from the AVATAR's artifacts, not from --model. A single
    # server fronts a mixed data/avatars/ directory, so connecting a wav2lip
    # avatar to a musetalk-configured server must load wav2lip (or, in remote
    # mode, switch the worker) instead of failing on a missing latents.pt.
    plugin = resolve_avatar_backend(avatar_id)          # 'musetalk'|'wav2lip'|...|'remote'
    ensure_model_loaded(plugin, avatar_id)

    if avatar_id not in global_avatars:
        global_avatars[avatar_id] = _load_avatar_payload(plugin, avatar_id)
    avatar_this = global_avatars[avatar_id]

    if ref_audio: #请求参数配置了参考音频
        opt_this.REF_FILE = ref_audio
        opt_this.REF_TEXT = ref_text
    custom_config=params.get('custom_config','') #动作编排配置
    if custom_config:
        opt_this.customopt = json.loads(custom_config)

    avatar_session = registry.create("avatar", plugin, opt=opt_this, model=model, avatar=avatar_this)
    return avatar_session

async def offer(request):
    return await rtc_manager.handle_offer(request)

async def whep(request):
    return await rtc_manager.handle_whep(request)

async def on_shutdown(app):
    await rtc_manager.shutdown()

async def download_record(request):
    sessionid = request.match_info.get('sessionid')
    if not sessionid:
        return web.Response(status=400, text="sessionid is required")
    
    record_file = os.path.join('data', 'record', f"{sessionid}.mp4")
    
    if os.path.exists(record_file):
        return web.FileResponse(record_file)
    else:
        return web.Response(status=404, text="Record not found")


def _announce_remote(opt):
    """Log what the remote GPU worker is, in the background, at startup.

    The worst failure mode of the host/worker split is silent: the server boots
    happily and the first browser connect is what discovers the worker is down,
    on the wrong revision, or CPU-only. Probing at startup turns that into a log
    line you see immediately. It runs in a daemon thread with a short timeout so
    a dead box never delays the HTTP listener.
    """
    from remote import revision, settings
    address, source = settings.effective()
    if not address:
        logger.error(
            '[remote] worker mode is remote but NO worker address is configured. '
            'Set it in /admin.html (Remote GPU Worker) or in config.yaml. '
            'Sessions will fail until then.')
        return
    local_rev = revision.current_revision()
    logger.info('[remote] worker mode (local/remote toggle = remote); worker=%s '
                '(source: %s), local revision %s',
                address, source, local_rev or 'unknown')

    if settings.tunnel_enabled():
        from remote import tunnel
        # The worker binds loopback on the box, so the only route is the tunnel.
        # Started here (not lazily on first connect) so the browser never waits
        # on an SSH handshake, and supervised in the background so a dropped
        # link is repaired instead of silently killing every later session.
        tunnel.start_background_supervisor()

    def _probe():
        from remote.client import probe
        try:
            info = probe(address, timeout=6.0)
        except Exception as exc:  # noqa: BLE001
            logger.error('[remote] worker %s is NOT reachable: %s. Start it on the '
                         'box, or run ./deploy.sh here to redeploy and restart it.',
                         address, exc)
            return
        if info.get('device') != 'cuda':
            logger.warning('[remote] worker %s reports device=%r, not cuda -- '
                           'inference will be slow', address, info.get('device'))
        remote_rev = info.get('revision') or ''
        if not remote_rev:
            logger.warning('[remote] worker reports no revision (old build); run '
                           './deploy.sh to bring it up to date')
        elif not revision.matches(remote_rev, local_rev):
            logger.warning('[remote] worker serves %s but this checkout is %s -- '
                           'run ./deploy.sh to sync the box', remote_rev, local_rev)
        else:
            logger.info('[remote] worker %s ready: %s, revision %s',
                        address, info.get('detail') or info.get('device'), remote_rev)

    Thread(target=_probe, daemon=True).start()


def main():
    global rtc_manager, opt, model, load_avatar, load_model, warm_up
    # 解析命令行参数
    from config import parse_args
    opt = parse_args()

    # --remote_worker is the single knob for both inference and enhancement:
    # publish it as env so server/enhance_routes.py (no access to opt) and the
    # remote avatar plugin both pick it up.
    if getattr(opt, 'remote_worker', ''):
        os.environ['LT_REMOTE_WORKER'] = opt.remote_worker

    # The local/remote toggle. Order of precedence is saved UI value > env >
    # CLI, and the saved value lives in remote/settings.py; --worker_mode only
    # supplies the fallback, which is why it is recorded as a default rather
    # than written to the settings file.
    from remote import settings as remote_settings
    remote_settings.set_default_mode(getattr(opt, 'worker_mode', 'local'))
    if opt.model == 'remote':
        # Legacy shorthand: `model: remote` / `--model remote` predates the
        # toggle and still means "use the worker".
        remote_settings.set_default_mode('remote')

    # The configured model must agree with the toggle: a stale `model: remote`
    # in config.yaml must not keep creating RemoteAvatar sessions after the
    # toggle says local (and vice versa).
    if worker_remote():
        opt.model = 'remote'
    elif opt.model == 'remote':
        opt.model = getattr(opt, 'remote_backend', '') or 'musetalk'
    logger.info('[model] worker mode: %s (GPU work runs %s)',
                remote_settings.worker_mode(),
                'on the remote box' if worker_remote() else 'locally')

    _write_pidfile()
    atexit.register(_remove_pidfile)

    # ─── 加载 avatar 插件（触发 @register 注册）──────────────────────
    # Import EVERY backend module, not just the configured one: a connect may
    # target an avatar belonging to a different backend, and registry.create()
    # needs that class registered before it can instantiate it.
    import importlib
    _available = []
    for _backend_name_, _module in _AVATAR_MODULES.items():
        try:
            importlib.import_module(_module)
            _available.append(_backend_name_)
        except Exception:
            # A backend whose weights/deps are missing must not stop the server
            # from serving the ones that do work. Log and move on; the connect
            # path will surface the failure if that avatar is requested.
            logger.warning('[model] backend %r unavailable (%s); its avatars '
                           'cannot be loaded', _backend_name_, _module, exc_info=True)
    # Legacy module-level handles kept for any external caller; the real
    # dispatch goes through _bind_backend()/resolve_avatar_backend().
    if opt.model in _available:
        load_model, load_avatar, warm_up = _bind_backend(opt.model)
    else:
        logger.error('[model] configured backend %r failed to import — set '
                     '--model to one of %s', opt.model, _available)
    logger.info(opt)

    # ─── Lazy model loading: nothing is loaded here. The first connect to any
    # avatar triggers ensure_model_loaded() (weights + default avatar + warmup).
    # ─── Manual unload/restart live in unload_model() / restart_app() below and
    # are exposed on /api/model/* and /api/app/restart (server/routes.py).
    if opt.skip_model:
        logger.info('[--skip-model] kept for compatibility; model loading is '
                    'lazy-on-first-connect by default now.')

    # model=remote means this process is a HOST for the GPU box: say so at
    # startup instead of letting the first browser connect discover it.
    if opt.model == 'remote':
        _announce_remote(opt)

    # init rtc manager
    session_manager.set_max_session(opt.max_session)
    session_manager.init_builder(build_avatar_session)
    rtc_manager = RTCManager(opt)
    # share avatar_sessions (RTCManager handles it but routes.py expects it)

    # 虚拟摄像头或 RTMP 模式：后台渲染线程在模型就绪后启动，不阻塞 HTTP 绑定
    if opt.transport == 'virtualcam' or opt.transport == 'rtmp':
        def _start_virtualcam():
            # build_avatar_session() waits on model_ready_event internally.
            session_manager.add_session('0', build_avatar_session('0', {}))
            rendthrd = Thread(target=session_manager.get_session('0').render, args=(thread_quit,))
            rendthrd.start()
        thread_quit = Event()
        Thread(target=_start_virtualcam, daemon=True).start()
        if opt.transport == 'virtualcam':
            logger.info("[VirtualCam] Virtual camera output enabled - digital human will be rendered to virtual camera")

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync["llm_response"] = llm_response
    appasync["opt"] = opt
    appasync["rtc_manager"] = rtc_manager

    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_get("/record/{sessionid}", download_record)

    # 注册 server/routes.py 中的通用 API 路由
    setup_routes(appasync)

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    # /whep 注册在 CORS 之后：自行管理 OPTIONS，避免与 aiohttp_cors 冲突
    whep_resource = appasync.router.add_resource('/whep')
    whep_resource.add_route('POST', whep)
    whep_resource.add_route('OPTIONS', lambda _: web.Response(status=200))

    logger.info('start http server; http://<serverip>:'+str(opt.listenport))
    # logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        try:
            loop.run_until_complete(site.start())
        except OSError as e:
            logger.error('[startup] cannot bind 0.0.0.0:%s — %s', opt.listenport, e)
            logger.error('[startup] another process is holding the port; '
                         'use ./restart.sh to stop it and relaunch')
            loop.run_until_complete(runner.cleanup())
            loop.close()
            _remove_pidfile()
            raise SystemExit(1)

        # ─── Graceful shutdown: SIGTERM/SIGINT stop sessions, close the HTTP
        # listener and return from run_forever, so restart.sh can bring the
        # process down quickly instead of waiting for a SIGKILL.
        shutting_down = False

        async def _graceful_shutdown(signame):
            nonlocal shutting_down
            if shutting_down:
                return
            shutting_down = True
            logger.info('[shutdown] %s received — stopping sessions and HTTP server', signame)
            try:
                await runner.cleanup()
            except Exception:
                logger.exception('[shutdown] cleanup failed')
            finally:
                loop.stop()

        def _handle_signal(signame):
            logger.info('[shutdown] %s received', signame)
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(_graceful_shutdown(signame)))

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _handle_signal, sig.name)
            except (NotImplementedError, RuntimeError):
                # Non-main thread or unsupported platform: fall back to the
                # default handler (KeyboardInterrupt exits with a traceback).
                logger.debug('[shutdown] signal handler for %s not installed', sig.name)

        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(rtc_manager.handle_rtcpush(push_url, str(k)))
        loop.run_forever()
        loop.close()
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()


# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    load_dotenv()  # Load environment variables from .env file, if it exists
    main()
    
    
    
