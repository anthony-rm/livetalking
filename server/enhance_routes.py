###############################################################################
#  Video enhancement API — background jobs around tools/enhance (RVM matting,
#  lighting + GFPGAN face restore, vertical composition, background swap).
#
#  Jobs are SERIALIZED on one worker: the enhancement stack (RVM + GFPGAN on
#  MPS/CUDA) is heavy enough that two concurrent jobs would thrash. Swap
#  (recompose) jobs share the same queue for the same reason.
#
#  Artifacts per job live in data/enhance/<task_id>/:
#    source.<ext>   uploaded or referenced input clip
#    enhanced.mp4   enhanced vertical result
#    matte/         person-matte PNGs + meta.json
#    person/        person color PNGs (same geometry as matte)
#    job.json       options + result metadata
###############################################################################

import json
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web

from tools.enhance import pipeline
from utils.logger import logger

ENHANCE_DIR = os.path.abspath("./data/enhance")

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def json_ok(data=None):
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(content_type="application/json",
                        text=json.dumps(body))


def json_error(msg: str, code: int = -1):
    return web.Response(content_type="application/json",
                        text=json.dumps({"code": code, "msg": str(msg)}))


class EnhanceJob:
    def __init__(self, task_id: str, params: dict, workdir: str):
        self.task_id = task_id
        self.params = params
        self.workdir = workdir
        self.status = "pending"          # pending, running, completed, failed, cancelled
        self.progress = 0
        self.phase = "queued"
        self.error_msg = ""
        self.result = {}
        self.cancel_requested = False
        self.start_time = time.time()
        self.end_time = None

    _PUBLIC_PARAMS = ("background", "fit", "width", "height", "strength",
                      "face_restore", "upscale", "max_frames", "source_job")

    def to_dict(self):
        d = {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "phase": self.phase,
            "error_msg": self.error_msg,
            "params": {k: self.params[k] for k in self._PUBLIC_PARAMS
                       if k in self.params},
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (self.end_time or time.time()) - self.start_time,
            "has_result": os.path.isfile(self.output_path),
        }
        # Chained avatar creation: the UI links an enhance job to the genavatar
        # task it spawned (and a reload of the page rediscover it from here).
        if self.params.get("avatar"):
            d["avatar"] = {k: v for k, v in self.params["avatar"].items()
                           if k in ("model", "avatar_id")}
            if self.params.get("avatar_task_id"):
                d["avatar_task_id"] = self.params["avatar_task_id"]
            if self.params.get("avatar_error"):
                d["avatar_error"] = self.params["avatar_error"]
        if self.result:
            d["result"] = self.result
        return d

    @property
    def output_path(self) -> str:
        return os.path.join(self.workdir, "enhanced.mp4")

    @property
    def source_path(self) -> str:
        return self.params.get("video_path") or ""


class EnhanceJobManager:
    def __init__(self, max_workers: int = 1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, EnhanceJob] = {}
        self.lock = threading.Lock()

    # -- job bookkeeping -----------------------------------------------------
    def create_job(self, params: dict) -> EnhanceJob:
        task_id = str(uuid.uuid4())[:8]
        workdir = os.path.join(ENHANCE_DIR, task_id)
        os.makedirs(workdir, exist_ok=True)
        # Own the artifact layout here, before submitting, so the worker never
        # sees a half-configured job (a recompose job reuses an existing matte).
        if not params.get("recompose_from_matte"):
            params.setdefault("matte_out", os.path.join(workdir, "matte"))
            params.setdefault("person_out", os.path.join(workdir, "person"))
        job = EnhanceJob(task_id, params, workdir)
        with self.lock:
            self.jobs[task_id] = job
        self.executor.submit(self._run, job)
        return job

    def get(self, task_id: str) -> EnhanceJob:
        return self.jobs.get(task_id)

    def list_jobs(self):
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.start_time,
                          reverse=True)
        return [j.to_dict() for j in jobs]

    def request_cancel(self, task_id: str) -> bool:
        job = self.get(task_id)
        if job is None:
            return False
        if job.status in ("pending", "running"):
            job.cancel_requested = True
            return True
        return False

    # -- execution -----------------------------------------------------------
    def _run(self, job: EnhanceJob) -> None:
        job.status = "running"
        job.phase = "starting"
        params = job.params
        try:
            def progress(pct: int, phase: str) -> None:
                job.progress = max(0, min(100, pct))
                job.phase = phase

            def cancelled() -> bool:
                return job.cancel_requested

            if params.get("recompose_from_matte"):
                # Recompose is cheap CPU work (no GPU models): always local.
                job.phase = "recompose"
                meta_path = os.path.join(params["matte_out"], "meta.json")
                with open(meta_path, encoding="utf-8") as fh:
                    fps = json.load(fh).get("fps", 30.0)
                result = pipeline.recompose_video(
                    params["matte_out"], job.output_path,
                    person_dir=params.get("person_out"),
                    background=params.get("background", pipeline.DEFAULT_BG),
                    fps=fps, progress_callback=progress)
            else:
                # Heavy path (RVM + Real-ESRGAN + GFPGAN): runs here, on the
                # machine with the GPU. The remote variant (one RPC for the
                # whole job) is gone with the worker split.
                result = pipeline.enhance_video(
                    params["video_path"], job.output_path,
                    width=params.get("width", pipeline.DEFAULT_WIDTH),
                    height=params.get("height", pipeline.DEFAULT_HEIGHT),
                    background=params.get("background", pipeline.DEFAULT_BG),
                    fit=params.get("fit", "contain"),
                    strength=params.get("strength", 0.5),
                    face_restore=params.get("face_restore", True),
                    upscale=params.get("upscale", 1),
                    max_frames=params.get("max_frames"),
                    matte_out=params.get("matte_out"),
                    person_out=params.get("person_out"),
                    progress_callback=progress, should_cancel=cancelled)

            job.result = result
            job.status = "cancelled" if job.cancel_requested else "completed"
            job.progress = 100
            if job.status == "completed":
                self._chain_avatar(job)
            self._write_job_meta(job)
            logger.info(f"[enhance] job {job.task_id} {job.status}")
        except pipeline.Cancelled:
            job.status = "cancelled"
            job.phase = "cancelled"
            self._write_job_meta(job)
            logger.info(f"[enhance] job {job.task_id} cancelled")
        except Exception as e:  # noqa: BLE001
            job.status = "failed"
            job.error_msg = str(e)
            logger.exception(f"[enhance] job {job.task_id} failed")
        finally:
            job.end_time = time.time()

    def _chain_avatar(self, job: EnhanceJob) -> None:
        """Hand a finished clip to genavatar when the job was submitted with an
        avatar spec. Failures are recorded on the job, never on genavatar's."""
        spec = job.params.get("avatar") or {}
        if not spec:
            return
        if not os.path.isfile(job.output_path):
            job.params["avatar_error"] = "enhanced video is missing"
            return
        try:
            from server.avatar_routes import enqueue_avatar_task
            job.params["avatar_task_id"] = enqueue_avatar_task(
                spec["model"], spec["avatar_id"], spec.get("params") or {},
                video_path=job.output_path, source_job=job.task_id)
            logger.info(f"[enhance] job {job.task_id} → avatar task "
                        f"{job.params['avatar_task_id']} "
                        f"({spec['model']} → {spec['avatar_id']})")
        except Exception as e:  # noqa: BLE001
            job.params["avatar_error"] = str(e)
            logger.exception(f"[enhance] job {job.task_id} avatar chain failed")

    def _write_job_meta(self, job: EnhanceJob) -> None:
        try:
            with open(os.path.join(job.workdir, "job.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(job.to_dict(), fh, indent=2, default=str)
        except Exception:  # noqa: BLE001
            logger.exception("[enhance] failed writing job.json")


enhance_manager = EnhanceJobManager()


# ─── request helpers ────────────────────────────────────────────────────────
def _bool(v, default=False):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


async def _save_upload(part) -> str:
    ext = os.path.splitext(part.filename or "")[1].lower() or ".mp4"
    if ext not in VIDEO_EXTS:
        raise ValueError(f"unsupported video extension: {ext}")
    workdir = os.path.join(ENHANCE_DIR, "uploads")
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, f"{str(uuid.uuid4())[:8]}{ext}")
    with open(path, "wb") as f:
        while True:
            chunk = await part.read_chunk()
            if not chunk:
                break
            f.write(chunk)
    return path


def _job_or_none(task_id: str) -> EnhanceJob:
    job = enhance_manager.get(task_id)
    if job is None:
        raise LookupError(f"task not found: {task_id}")
    return job


def _safe_source_copy(job: EnhanceJob) -> str:
    """Copy the source clip into the job workdir so sidecars stay self-contained."""
    src = job.source_path
    if not src or not os.path.isfile(src):
        raise ValueError(f"source video not found: {src}")
    ext = os.path.splitext(src)[1].lower() or ".mp4"
    dst = os.path.join(job.workdir, f"source{ext}")
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)
        job.params["video_path"] = dst
    return dst


# ─── handlers ───────────────────────────────────────────────────────────────
def _parse_avatar_spec(params: dict) -> dict:
    """Extract the optional enhance → genavatar chain from request params.

    Accepts either a nested `avatar` object (JSON body) or the flat multipart
    fields avatar_model / avatar_id / avatar_params (a JSON object string).
    Raises ValueError on a malformed spec so a typo fails fast instead of
    silently skipping the avatar step.
    """
    from server.avatar_routes import AVATAR_GENERATORS, is_supported_model

    spec = params.get("avatar")
    if isinstance(spec, str):
        spec = json.loads(spec) if spec.strip() else None
    if not spec:
        model = params.get("avatar_model")
        avatar_id = params.get("avatar_id")
        if not model and not avatar_id:
            return {}
        gen_params = params.get("avatar_params")
        if isinstance(gen_params, str):
            gen_params = json.loads(gen_params) if gen_params.strip() else {}
        spec = {"model": model, "avatar_id": avatar_id,
                "params": gen_params or {}}
    if not isinstance(spec, dict):
        raise ValueError("avatar must be an object")

    model = (spec.get("model") or "").strip()
    avatar_id = (spec.get("avatar_id") or "").strip()
    if not model or not avatar_id:
        raise ValueError("avatar.model and avatar.avatar_id are required")
    if not is_supported_model(model):
        raise ValueError(f"avatar.model must be one of {list(AVATAR_GENERATORS)}")
    gen_params = spec.get("params") or {}
    if not isinstance(gen_params, dict):
        raise ValueError("avatar.params must be an object")
    return {"model": model, "avatar_id": avatar_id, "params": gen_params}


async def create_enhance_task(request):
    """POST /api/enhance/task — video_file (upload) or video_path + options.

    Optional `avatar` spec (or avatar_model / avatar_id / avatar_params): when
    present, a genavatar task is enqueued automatically as soon as the enhance
    job completes — the enhance → avatar chain of the merged studio page.
    """
    try:
        params: dict = {}
        if request.content_type == "multipart/form-data":
            reader = await request.multipart()
            video_path = None
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name == "video_file":
                    video_path = await _save_upload(part)
                else:
                    params[part.name] = await part.text()
            if video_path:
                params["video_path"] = video_path
        else:
            params = await request.json()

        video_path = params.get("video_path")
        if not video_path:
            return json_error("video_file or video_path is required")
        if not os.path.isfile(video_path):
            return json_error(f"video not found: {video_path}")

        avatar_spec = _parse_avatar_spec(params)

        job_params = {
            "video_path": os.path.abspath(video_path),
            "width": int(params.get("width", pipeline.DEFAULT_WIDTH)),
            "height": int(params.get("height", pipeline.DEFAULT_HEIGHT)),
            "background": params.get("background", pipeline.DEFAULT_BG),
            "fit": params.get("fit", "contain"),
            "strength": float(params.get("strength", 0.5)),
            "face_restore": _bool(params.get("face_restore"), True),
            "upscale": int(params.get("upscale", 1)),
            "max_frames": int(params["max_frames"]) if params.get("max_frames") else None,
        }
        if avatar_spec:
            job_params["avatar"] = avatar_spec
        job = enhance_manager.create_job(params=job_params)
        try:
            _safe_source_copy(job)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[enhance] could not snapshot source: {e}")
        return json_ok(data={"task_id": job.task_id})
    except Exception as e:  # noqa: BLE001
        logger.exception("create_enhance_task error")
        return json_error(str(e))


async def get_enhance_task(request):
    try:
        job = _job_or_none(request.match_info["task_id"])
        d = job.to_dict()
        if os.path.isfile(os.path.join(job.workdir, "source.mp4")):
            d["source"] = "/api/enhance/task/%s/video?kind=source" % job.task_id
        if job.status in ("completed", "cancelled") and os.path.isfile(job.output_path):
            d["video_url"] = f"/api/enhance/task/{job.task_id}/video?kind=enhanced"
            d["download_url"] = d["video_url"] + "&download=1"
        return json_ok(data=d)
    except LookupError as e:
        return json_error(str(e), code=404)
    except Exception as e:  # noqa: BLE001
        logger.exception("get_enhance_task error")
        return json_error(str(e))


async def list_enhance_tasks(request):
    try:
        return json_ok(data={"tasks": enhance_manager.list_jobs()})
    except Exception as e:  # noqa: BLE001
        return json_error(str(e))


async def cancel_enhance_task(request):
    try:
        ok = enhance_manager.request_cancel(request.match_info["task_id"])
        if not ok:
            return json_error("task not found or not cancellable")
        return json_ok(data={"cancelling": True})
    except Exception as e:  # noqa: BLE001
        return json_error(str(e))


async def recompose_enhance_task(request):
    """POST /api/enhance/task/{id}/recompose — swap background from sidecars."""
    try:
        job = _job_or_none(request.match_info["task_id"])
        body = await request.json()
        background = body.get("background", pipeline.DEFAULT_BG)
        matte_dir = job.params.get("matte_out")
        if not matte_dir or not os.path.isfile(os.path.join(matte_dir, "meta.json")):
            return json_error("this job has no matte sidecar; run an enhance job "
                              "with sidecars enabled first")
        person_dir = job.params.get("person_out")
        new_job = enhance_manager.create_job(params={
            "recompose_from_matte": True,
            "matte_out": matte_dir,
            "person_out": person_dir,
            "background": background,
            "video_path": job.source_path,
            "source_job": job.task_id,
        })
        return json_ok(data={"task_id": new_job.task_id, "recompose_of": job.task_id})
    except LookupError as e:
        return json_error(str(e), code=404)
    except Exception as e:  # noqa: BLE001
        logger.exception("recompose_enhance_task error")
        return json_error(str(e))


async def create_avatar_from_enhanced(request):
    """POST /api/enhance/task/{id}/avatar — hand the enhanced clip to genavatar."""
    try:
        from server.avatar_routes import (AVATAR_GENERATORS, enqueue_avatar_task,
                                          is_supported_model)
        job = _job_or_none(request.match_info["task_id"])
        if job.status != "completed" or not os.path.isfile(job.output_path):
            return json_error("enhance job is not completed yet")
        body = await request.json()
        model = body.get("model")
        avatar_id = body.get("avatar_id")
        if not model or not avatar_id:
            return json_error("model and avatar_id are required")
        if not is_supported_model(model):
            return json_error(f"model must be one of {list(AVATAR_GENERATORS)}")
        task_id = enqueue_avatar_task(model, avatar_id,
                                      body.get("params") or {},
                                      video_path=job.output_path,
                                      source_job=job.task_id)
        # Remember the link on the job so a page reload keeps showing it.
        job.params["avatar"] = {"model": model, "avatar_id": avatar_id,
                                "params": body.get("params") or {}}
        job.params["avatar_task_id"] = task_id
        return json_ok(data={"avatar_task_id": task_id,
                             "video": job.output_path})
    except LookupError as e:
        return json_error(str(e), code=404)
    except Exception as e:  # noqa: BLE001
        logger.exception("create_avatar_from_enhanced error")
        return json_error(str(e))


async def serve_enhance_video(request):
    """GET /api/enhance/task/{id}/video?kind=enhanced|source&download=1"""
    try:
        job = _job_or_none(request.match_info["task_id"])
        kind = request.query.get("kind", "enhanced")
        if kind == "source":
            src = job.params.get("video_path") or ""
            if not src or not os.path.isfile(src):
                return json_error("source video unavailable", code=404)
            path = src
        else:
            path = job.output_path
            if not os.path.isfile(path):
                return json_error("result not ready", code=404)
        if not os.path.abspath(path).startswith(ENHANCE_DIR) and kind != "source":
            return json_error("invalid path", code=400)
        headers = {}
        if _bool(request.query.get("download")):
            headers["Content-Disposition"] = \
                f'attachment; filename="{job.task_id}_{kind}.mp4"'
        return web.FileResponse(path, headers=headers)
    except LookupError as e:
        return json_error(str(e), code=404)
    except Exception as e:  # noqa: BLE001
        logger.exception("serve_enhance_video error")
        return json_error(str(e))


async def delete_enhance_task(request):
    try:
        job = _job_or_none(request.match_info["task_id"])
        if job.status in ("pending", "running"):
            return json_error("task is still running; cancel it first")
        with enhance_manager.lock:
            enhance_manager.jobs.pop(job.task_id, None)
        if os.path.isdir(job.workdir):
            shutil.rmtree(job.workdir, ignore_errors=True)
        return json_ok(data={"deleted": job.task_id})
    except LookupError as e:
        return json_error(str(e), code=404)
    except Exception as e:  # noqa: BLE001
        return json_error(str(e))


def setup_enhance_routes(app):
    app.router.add_post("/api/enhance/task", create_enhance_task)
    app.router.add_get("/api/enhance/tasks", list_enhance_tasks)
    app.router.add_get("/api/enhance/task/{task_id}", get_enhance_task)
    app.router.add_post("/api/enhance/task/{task_id}/recompose", recompose_enhance_task)
    app.router.add_post("/api/enhance/task/{task_id}/avatar", create_avatar_from_enhanced)
    app.router.add_post("/api/enhance/task/{task_id}/cancel", cancel_enhance_task)
    app.router.add_get("/api/enhance/task/{task_id}/video", serve_enhance_video)
    app.router.add_delete("/api/enhance/task/{task_id}", delete_enhance_task)