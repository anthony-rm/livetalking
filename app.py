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
def _do_load_models():
    """Load the weights for the configured --model backend."""
    global model
    if opt.model == 'musetalk':
        model = load_model()
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model)
    elif opt.model == 'wav2lip':
        model = load_model("./models/wav2lip.pth")
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model, 256)
    elif opt.model == 'ultralight':
        model = load_model(opt)
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, global_avatars[opt.avatar_id], 160)
    elif opt.model == 'remote':
        # The GPU lives on the Windows worker; load_model() connects + tells the
        # worker to load the backend weights, load_avatar() ships the inference
        # tensors, warm_up() primes the remote CUDA path.
        model = load_model(opt)
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model)


def ensure_model_loaded():
    """Lazy-load the avatar model on first connect (thread-safe).

    Concurrent callers wait on model_ready_event instead of loading twice.
    """
    global model_state
    if model_state == 'ready':
        return
    with model_state_lock:
        if model_state == 'ready':
            return
        if model_state == 'loading':
            # Another thread is already loading: wait for it to finish.
            model_ready_event.wait()
            if model_state != 'ready':
                raise RuntimeError('avatar model failed to load — check server logs')
            return
        model_state = 'loading'
        model_ready_event.clear()
        try:
            _do_load_models()
            model_state = 'ready'
            logger.info('[model] avatar model loaded and warmed up (lazy load)')
        except Exception:
            model_state = 'unloaded'
            logger.exception('[model] model load/warmup failed')
            raise
        finally:
            model_ready_event.set()


def unload_model(force: bool = False) -> str:
    """Free model weights + cached avatars. Refuses while sessions are active
    unless force=True, which disconnects them first."""
    global model, model_state
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
        if opt and opt.model == 'remote':
            try:
                from avatars.remote_avatar import unload_remote
                unload_remote()
            except Exception:  # noqa: BLE001
                logger.warning('[model] remote worker unload failed', exc_info=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_state = 'unloaded'
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
    ensure_model_loaded()
    opt_this = copy.deepcopy(opt)
    opt_this.sessionid = sessionid

    avatar_id = params.get('avatar',opt.avatar_id) 
    opt_this.avatar_id = avatar_id
    ref_audio = params.get('refaudio','') #音色
    ref_text = params.get('reftext','')
    if (avatar_id and avatar_id != opt.avatar_id):
        # Avoid reloading if already cached globally
        if avatar_id not in global_avatars:
            global_avatars[avatar_id] = load_avatar(avatar_id)
        avatar_this = global_avatars[avatar_id]
    else:
        # Default avatar loaded at startup
        avatar_this = global_avatars.get(opt.avatar_id)
    if ref_audio: #请求参数配置了参考音频
        opt_this.REF_FILE = ref_audio
        opt_this.REF_TEXT = ref_text
    custom_config=params.get('custom_config','') #动作编排配置
    if custom_config:
        opt_this.customopt = json.loads(custom_config)

    avatar_session = registry.create("avatar", opt.model, opt=opt_this, model=model, avatar=avatar_this)
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

    _write_pidfile()
    atexit.register(_remove_pidfile)

    # ─── 加载 avatar 插件（触发 @register 注册）──────────────────────
    _avatar_modules = {
        'musetalk':   'avatars.musetalk_avatar',
        'wav2lip':    'avatars.wav2lip_avatar',
        'ultralight': 'avatars.ultralight_avatar',
        'remote':     'avatars.remote_avatar',
    }
    import importlib
    avatar_mod = importlib.import_module(_avatar_modules[opt.model])
    load_model = avatar_mod.load_model
    load_avatar = avatar_mod.load_avatar
    warm_up = avatar_mod.warm_up
    logger.info(opt)

    # ─── Lazy model loading: nothing is loaded here. The first connect to any
    # avatar triggers ensure_model_loaded() (weights + default avatar + warmup).
    # ─── Manual unload/restart live in unload_model() / restart_app() below and
    # are exposed on /api/model/* and /api/app/restart (server/routes.py).
    if opt.skip_model:
        logger.info('[--skip-model] kept for compatibility; model loading is '
                    'lazy-on-first-connect by default now.')

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
    
    
    
