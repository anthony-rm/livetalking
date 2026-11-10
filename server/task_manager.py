import uuid
import time
import threading
import json
import os
import requests
import cv2
from concurrent.futures import ThreadPoolExecutor
from utils.logger import logger

class AvatarTask:
    def __init__(self, task_id, model_type, avatar_id, params, notify_url=None):
        self.task_id = task_id
        self.model_type = model_type
        self.avatar_id = avatar_id
        self.params = params
        self.status = "pending"  # pending, running, completed, failed
        self.progress = 0
        self.error_msg = ""
        self.notify_url = notify_url
        self.start_time = time.time()
        self.end_time = None

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "model_type": self.model_type,
            "avatar_id": self.avatar_id,
            "status": self.status,
            "progress": self.progress,
            "error_msg": self.error_msg,
            "notify_url": self.notify_url,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (self.end_time or time.time()) - self.start_time,
            # Set when this avatar was generated from an enhance job's output.
            "source_job": self.params.get("source_job"),
        }

class TaskManager:
    def __init__(self, max_workers=1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks = {}
        self.lock = threading.Lock()

    def add_task(self, model_type, avatar_id, params, task_id=None, notify_url=None):
        if task_id is None:
            task_id = str(uuid.uuid4())

        task = AvatarTask(task_id, model_type, avatar_id, params, notify_url)
        with self.lock:
            self.tasks[task_id] = task

        self._notify(task)
        self.executor.submit(self._run_task, task_id)
        return task_id

    def get_task(self, task_id):
        with self.lock:
            return self.tasks.get(task_id)

    def delete_task(self, task_id):
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False, "Task not found"

            if task.status != "pending":
                return False, f"Task is in {task.status} state, cannot delete"

            del self.tasks[task_id]
            return True, "Task deleted"

    def list_tasks(self):
        with self.lock:
            return sorted([task.to_dict() for task in self.tasks.values()], key=lambda x: x['start_time'], reverse=True)

    def _run_task(self, task_id):
        task = self.get_task(task_id)
        if not task:
            return

        task.status = "running"
        self._notify(task)
        logger.info(f"Starting task {task_id} for model {task.model_type}, avatar {task.avatar_id}")

        try:
            def progress_callback(p):
                task.progress = p

            # Avatar creation is the heaviest GPU work in the app (per-frame VAE
            # encode + face parsing). When a worker is configured, build it
            # THERE; only the small artifact comes back. Falling back to local
            # generation would run for hours on MPS/CPU and is not what the
            # remote configuration promises.
            if self._should_use_remote():
                self._run_remote(task, progress_callback)
            elif task.model_type == "musetalk":
                from avatars.musetalk.genavatar import generate_avatar
                generate_avatar(
                    video_path=task.params['video_path'],
                    avatar_id=task.avatar_id,
                    save_path=task.params.get('save_path', './data/avatars'),
                    bbox_shift=task.params.get('bbox_shift', 0),
                    extra_margin=task.params.get('extra_margin', 10),
                    parsing_mode=task.params.get('parsing_mode', 'jaw'),
                    version=task.params.get('version', 'v15'),
                    progress_callback=progress_callback
                )
            elif task.model_type == "wav2lip":
                from avatars.wav2lip.genavatar import generate_avatar
                generate_avatar(
                    video_path=task.params['video_path'],
                    avatar_id=task.avatar_id,
                    save_path=task.params.get('save_path', './data/avatars'),
                    img_size=task.params.get('img_size', 96),
                    pads=task.params.get('pads', [0, 10, 0, 0]),
                    nosmooth=task.params.get('nosmooth', False),
                    face_det_batch_size=task.params.get('face_det_batch_size', 16),
                    progress_callback=progress_callback
                )
            else:
                raise ValueError(f"Unsupported model type: {task.model_type}")

            task.status = "completed"
            task.progress = 100
            self._notify(task)
            logger.info(f"Task {task_id} completed successfully")
        except Exception as e:
            task.status = "failed"
            task.error_msg = str(e)
            self._notify(task)
            logger.error(f"Task {task_id} failed: {e}")
        finally:
            task.end_time = time.time()

    def _should_use_remote(self) -> bool:
        """True when avatar creation should run on the GPU worker.

        Gated on the local/remote toggle AND an effective address, so a
        configured-but-toggled-off box never quietly moves a multi-hour job to
        the worker (or the reverse). Import failures (grpc missing) degrade to
        local generation rather than failing every task.
        """
        try:
            from remote.settings import remote_active
            return remote_active()
        except Exception:  # noqa: BLE001
            logger.warning("[avatar] remote mode lookup failed; running locally",
                           exc_info=True)
            return False

    def _run_remote(self, task, progress_callback) -> None:
        """Generate `task` on the worker and materialise its artifact locally.

        The worker returns a zip of exactly what genavatar wrote, so the host
        ends up with the same `data/avatars/<id>/` tree as a local run — every
        downstream consumer (inference, paste_back, /api/avatars) is unchanged.
        """
        import io
        import shutil
        import zipfile

        from remote.client import RemoteClient, RemoteWorkerError
        from remote.settings import effective_address

        address = effective_address()
        progress_callback(2)
        source_path = task.params["video_path"]
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"source clip not found: {source_path}")

        # Ship the clip bytes, not a path: the box has no access to this
        # machine's filesystem. A typical clip is a few MB, far cheaper than
        # syncing a directory.
        with open(source_path, "rb") as fh:
            source_bytes = fh.read()

        params = {k: v for k, v in task.params.items()
                  if k not in ("video_path", "save_path", "source_job")}
        progress_callback(5)

        client = RemoteClient(address)
        try:
            archive = client.generate_avatar(
                task.model_type, task.avatar_id,
                source_bytes=source_bytes,
                filename=os.path.basename(source_path),
                params=params)
        finally:
            client.close()
        progress_callback(90)

        save_path = task.params.get("save_path", "./data/avatars")
        dest = os.path.join(save_path, task.avatar_id)
        tmp = dest + ".tmp-remote"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            zf.extractall(tmp)
        # Replace atomically: a partial extract must never look like an avatar.
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        os.replace(tmp, dest)
        task.params["generated_on"] = "remote"
        logger.info(f"[avatar] task {task.task_id} generated on {address} "
                    f"-> {dest}")
        progress_callback(99)

    def _notify(self, task):
        if not task.notify_url:
            return

        try:
            payload = task.to_dict()
            logger.info(f"Sending notification for task {task.task_id} to {task.notify_url}")
            requests.post(task.notify_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send notification for task {task.task_id}: {e}")

task_manager = TaskManager()
