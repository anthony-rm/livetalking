import os
import json
import uuid
from typing import Optional

from aiohttp import web
from server.task_manager import task_manager
from utils.logger import logger

# ─── Avatar generator registry ──────────────────────────────────────────────
# Single source of truth for what the studio can generate AND which knobs each
# backend exposes. The web UI renders its model select and its "Advanced" inputs
# from GET /api/avatar/models — never from hardcoded markup, so adding a backend
# here is all it takes for the page to offer it.
AVATAR_GENERATORS = {
    "wav2lip": {
        "label": "Wav2Lip",
        "modules": "avatars/wav2lip/genavatar.py",
        "params": [
            {"name": "img_size", "label": "Image size", "type": "int",
             "default": 256, "min": 32, "max": 512,
             "hint": "Side of the square face crop fed to the model."},
            {"name": "pads", "label": "Pads (top bottom left right)",
             "type": "text", "default": "0 10 0 0",
             "hint": "Padding around the detected face box, pixels."},
            {"name": "face_det_batch_size", "label": "Face detect batch size",
             "type": "int", "default": 16, "min": 1, "max": 64},
            {"name": "nosmooth", "label": "Disable bbox smoothing",
             "type": "bool", "default": False},
        ],
    },
    "musetalk": {
        "label": "MuseTalk",
        "modules": "avatars/musetalk/genavatar.py",
        "params": [
            {"name": "bbox_shift", "label": "BBox shift", "type": "int",
             "default": 0, "min": -50, "max": 50,
             "hint": "Moves the mouth region up/down; negative reveals more chin."},
            {"name": "extra_margin", "label": "Extra margin", "type": "int",
             "default": 10, "min": 0, "max": 100},
            {"name": "parsing_mode", "label": "Face parsing mode",
             "type": "select", "default": "jaw",
             "options": ["jaw", "neck", "raw"]},
            {"name": "version", "label": "MuseTalk version", "type": "select",
             "default": "v15", "options": ["v15", "v1"]},
        ],
    },
}

DEFAULT_GENERATOR = "wav2lip"


def avatar_generators():
    """Registry as JSON for the studio UI."""
    return [{"id": name,
             "label": spec["label"],
             "params": spec["params"]}
            for name, spec in AVATAR_GENERATORS.items()]


def is_supported_model(model) -> bool:
    return model in AVATAR_GENERATORS


def json_ok(data=None):
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )

def json_error(msg: str, code: int = -1):
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )

def _coerce_param(spec: dict, raw, context: str = ""):
    """Turn a raw form/JSON value into the type the generator expects."""
    kind = spec.get("type", "text")
    default = spec.get("default")
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return default
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if kind == "int":
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            raise ValueError(f"{spec['name']} must be an integer, got {raw!r}")
    if kind == "select":
        options = spec.get("options") or []
        value = str(raw).strip()
        if options and value not in options:
            raise ValueError(f"{spec['name']} must be one of {options}, got {value!r}")
        return value
    # Plain text: the pads field is space-separated, everything else verbatim.
    return str(raw).strip()


def build_task_params(model_type: str, raw_params: dict, video_path: str,
                      save_path: str = "./data/avatars") -> dict:
    """Validate/normalize generator arguments against the registry."""
    if not is_supported_model(model_type):
        raise ValueError(f"model must be one of {list(AVATAR_GENERATORS)}")
    spec = AVATAR_GENERATORS[model_type]
    task_params: dict = {"video_path": video_path, "save_path": save_path}
    for p in spec["params"]:
        task_params[p["name"]] = _coerce_param(p, raw_params.get(p["name"]),
                                              context=model_type)
    if model_type == "wav2lip":
        pads = task_params.get("pads")
        if isinstance(pads, str):
            try:
                task_params["pads"] = [int(x) for x in pads.split()]
            except ValueError:
                raise ValueError(f"pads must be four integers, got {pads!r}")
        if not isinstance(task_params.get("pads"), list) or len(task_params["pads"]) != 4:
            raise ValueError("pads must be four integers (top bottom left right)")
    return task_params


def enqueue_avatar_task(model_type: str, avatar_id: str, raw_params: dict,
                        video_path: str, notify_url: Optional[str] = None,
                        task_id: Optional[str] = None,
                        source_job: Optional[str] = None) -> str:
    """Create a genavatar task. Shared by the direct route and the
    enhance → avatar chain so both validate identically."""
    if not is_supported_model(model_type):
        raise ValueError(f"model must be one of {list(AVATAR_GENERATORS)}")
    params = build_task_params(model_type, raw_params, video_path)
    if source_job:
        params["source_job"] = source_job
    return task_manager.add_task(model_type, avatar_id, params,
                                 task_id=task_id, notify_url=notify_url)


async def get_avatar_models(request):
    """
    GET /api/avatar/models
    The generator registry: model ids, labels and their tunable parameters.
    """
    return json_ok(data={"models": avatar_generators(),
                         "default": DEFAULT_GENERATOR})


async def create_avatar_task(request):
    """
    POST /api/avatar/task
    Parameters: model, avatar_id, video_file (upload), video_path (local), ...
    """
    try:
        if request.content_type == 'multipart/form-data':
            reader = await request.multipart()
            params = {}
            video_path = None

            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name == 'video_file':
                    filename = part.filename
                    temp_dir = os.path.abspath('./data/tmp')
                    os.makedirs(temp_dir, exist_ok=True)
                    video_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{filename}")
                    with open(video_path, 'wb') as f:
                        while True:
                            chunk = await part.read_chunk()
                            if not chunk:
                                break
                            f.write(chunk)
                else:
                    value = await part.text()
                    params[part.name] = value

            if video_path:
                params['video_path'] = video_path
        else:
            params = await request.json()

        model_type = params.get('model')
        avatar_id = params.get('avatar_id')

        if not model_type or not avatar_id:
            return json_error("model and avatar_id are required")
        if not is_supported_model(model_type):
            return json_error(f"model must be one of {list(AVATAR_GENERATORS)}")

        if 'video_path' not in params:
            return json_error("video_file or video_path is required")

        data_path = './data/avatars'

        video_path = params['video_path']
        if not os.path.isabs(video_path):
            video_path = os.path.join(data_path, video_path)

        task_id = enqueue_avatar_task(
            model_type, avatar_id, params, video_path,
            notify_url=params.get('notifyurl'),
            task_id=params.get('task_id'))
        return json_ok(data={"task_id": task_id})

    except Exception as e:
        logger.exception("create_avatar_task error:")
        return json_error(str(e))

async def get_avatar_task_status(request):
    """
    GET /api/avatar/task/{task_id}
    """
    task_id = request.match_info.get('task_id')
    task = task_manager.get_task(task_id)
    if not task:
        return json_error("Task not found", code=404)

    return json_ok(data=task.to_dict())

async def list_avatar_tasks(request):
    """
    GET /api/avatar/tasks
    """
    tasks = task_manager.list_tasks()
    return json_ok(data={"tasks": tasks})

async def list_avatars(request):
    """
    GET /api/avatars
    列出 data/avatars 目录下所有已生成的 Avatar ID（按名称排序）。

    Only directories with a recognizable artifact layout are listed: the drop-in
    folder also holds .DS_Store/.gitkeep, and the connect page turns this list
    into a dropdown -- offering an entry that cannot possibly load is a bug the
    user only discovers on connect.
    """
    try:
        from avatars import detect as avatar_detect
        avatars_dir = os.path.abspath('./data/avatars')
        if not os.path.isdir(avatars_dir):
            return json_ok(data={"avatars": []})
        avatars = sorted(
            name for name in os.listdir(avatars_dir)
            if os.path.isdir(os.path.join(avatars_dir, name))
            and avatar_detect.detect_backend(name, root=avatars_dir)[0] is not None
        )
        return json_ok(data={"avatars": avatars})
    except Exception as e:
        logger.exception('list_avatars exception:')
        return json_error(str(e))

async def delete_avatar_task(request):
    """
    DELETE /api/avatar/task/{task_id}
    """
    task_id = request.match_info.get('task_id')
    success, msg = task_manager.delete_task(task_id)
    if not success:
        return json_error(msg)
    return json_ok(data={"msg": msg})

def setup_avatar_routes(app):
    app.router.add_post("/api/avatar/task", create_avatar_task)
    app.router.add_get("/api/avatar/task/{task_id}", get_avatar_task_status)
    app.router.add_delete("/api/avatar/task/{task_id}", delete_avatar_task)
    app.router.add_get("/api/avatar/tasks", list_avatar_tasks)
    app.router.add_get("/api/avatar/models", get_avatar_models)
    app.router.add_get("/api/avatars", list_avatars)