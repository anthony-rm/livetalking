"""Remote dispatch for the base-video enhance pipeline.

When ``LT_REMOTE_WORKER`` is set (or ``--remote_worker`` was passed), enhance
jobs run on the Windows/NVIDIA worker instead of locally: the Mac uploads the
source clip, the worker runs the full ``tools.enhance`` pipeline (RVM matting +
Real-ESRGAN + GFPGAN) on CUDA, and a zip with the enhanced mp4 + matte/person
sidecars comes back into the job's workdir. The existing job status/serving
routes work unchanged.
"""
from __future__ import annotations

import io
import os
import zipfile
from typing import Optional

from utils.logger import logger


def remote_address() -> Optional[str]:
    # Effective address: admin-UI value wins over the --remote_worker CLI flag
    # / LT_REMOTE_WORKER env var. See remote/settings.py.
    from remote.settings import effective_address
    return effective_address() or None


def remote_enabled() -> bool:
    return remote_address() is not None


def enhance_video_remote(job, params: dict, output_path: str,
                         matte_dir: Optional[str],
                         person_dir: Optional[str]) -> dict:
    """Run one enhance job on the remote worker.

    ``job`` is an EnhanceJob (uses task_id + cancel_requested). ``params`` are
    the pipeline options. Returns the pipeline result dict.
    """
    from remote.client import RemoteClient, RemoteWorkerError

    address = remote_address()
    client = RemoteClient(address)
    try:
        client.wait_ready(deadline=10.0)
        reply = client.enhance_video(
            params["video_path"],
            width=params.get("width", 1080),
            height=params.get("height", 1920),
            background=params.get("background", "#1E1E1E"),
            fit=params.get("fit", "contain"),
            strength=float(params.get("strength", 0.5)),
            face_restore=bool(params.get("face_restore", True)),
            upscale=int(params.get("upscale", 1)),
            max_frames=params.get("max_frames"),
            want_matte=matte_dir is not None,
            want_person=person_dir is not None,
        )
    finally:
        client.close()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if matte_dir:
        os.makedirs(matte_dir, exist_ok=True)
    if person_dir:
        os.makedirs(person_dir, exist_ok=True)

    result: dict = {}
    buf = io.BytesIO(reply.archive)
    with zipfile.ZipFile(buf) as zf:
        for name in zf.namelist():
            data = zf.read(name)
            if name == "enhanced.mp4":
                with open(output_path, "wb") as fh:
                    fh.write(data)
                continue
            if name == "meta.json":
                import json
                result = json.loads(data)
                continue
            # matte/00000000.png, person/00000000.png, ...
            root = name.split("/", 1)[0]
            if root == "matte" and matte_dir:
                dst = os.path.join(matte_dir, name[len("matte/"):])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as fh:
                    fh.write(data)
            elif root == "person" and person_dir:
                dst = os.path.join(person_dir, name[len("person/"):])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as fh:
                    fh.write(data)

    logger.info('[remote] enhance job %s: %s frames -> %s',
                getattr(job, "task_id", "?"), reply.frames, output_path)
    return result
