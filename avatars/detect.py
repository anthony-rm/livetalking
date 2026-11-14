"""Avatar artifact -> backend detection.

An avatar's backend is not stored anywhere authoritative; it is implied by the
files ``genavatar`` wrote. ``avator_info.json`` only exists for MuseTalk and is
also absent on older artifacts, so the directory layout is the ground truth:

    ultralight   ultralight.pth + face_imgs/ + full_imgs/ + coords.pkl
    musetalk     latents.pt + mask/ + mask_coords.pkl + full_imgs/ + coords.pkl
    ditto        ditto.json + f_s/ + x_s_info/ + full_imgs/ + coords.pkl
    wav2lip      face_imgs/ + full_imgs/ + coords.pkl

wav2lip and ultralight both have ``face_imgs/``, so ``ultralight.pth`` must be
checked before falling back to wav2lip. ditto shares ``full_imgs/`` and
``coords.pkl`` but not ``face_imgs/``; its ``ditto.json`` marker is checked
first, because a partial ditto tree would otherwise be misread.

Detection is read-only and cheap (a few
``os.path.exists`` calls): the connect path calls it before loading anything.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

# Backends the server can drive. Order matters only for documentation; the
# detection rules below are explicit.
KNOWN_BACKENDS = ("musetalk", "wav2lip", "ultralight", "ditto")

DEFAULT_AVATAR_ROOT = "./data/avatars"


def avatar_dir(avatar_id: str, root: str = DEFAULT_AVATAR_ROOT) -> str:
    return os.path.join(root, avatar_id)


def _has_images(path: str) -> bool:
    """True if `path` holds at least one image file (any common extension)."""
    try:
        for name in os.listdir(path):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                return True
    except OSError:
        return False
    return False


def detect_backend(avatar_id: str,
                   root: str | None = None) -> Tuple[Optional[str], str]:
    """Return ``(backend, reason)`` for an avatar directory.

    ``backend`` is None when the directory does not exist or its layout matches
    no known backend; ``reason`` always explains the decision so callers can put
    it in an error message instead of a bare "not found".

    ``root`` defaults to the module constant read at CALL time (not bound as a
    default argument) so tests and callers can redirect avatar discovery.
    """
    root = root or DEFAULT_AVATAR_ROOT
    path = avatar_dir(avatar_id, root)
    if not os.path.isdir(path):
        return None, f"avatar directory {path} does not exist"

    # Ditto first, and before any face_imgs/ test: a ditto artifact also ships
    # full_imgs/ + coords.pkl for paste-back, so without this rule it would read
    # as wav2lip and then fail on a missing face_imgs/ payload. ditto.json is
    # the explicit marker (it also carries the f_s dtype and render size the
    # loader needs), so its presence is authoritative.
    has_ditto_json = os.path.exists(os.path.join(path, "ditto.json"))
    if has_ditto_json:
        missing = [p for p in ("f_s", "x_s_info", "full_imgs")
                   if not os.path.exists(os.path.join(path, p))]
        if missing:
            return "ditto", (f"found ditto.json but missing {', '.join(missing)} "
                             f"— the artifact is incomplete")
        return "ditto", "found ditto.json + f_s/ + x_s_info/ + full_imgs/"

    # Ultralight first: it shares face_imgs/ with wav2lip.
    if os.path.exists(os.path.join(path, "ultralight.pth")):
        return "ultralight", "found ultralight.pth"

    has_latents = os.path.exists(os.path.join(path, "latents.pt"))
    has_mask = os.path.isdir(os.path.join(path, "mask"))
    if has_latents and has_mask:
        return "musetalk", "found latents.pt + mask/"
    if has_latents and not has_mask:
        # Half-built musetalk artifact: latents exist but the masks do not, so
        # paste_back cannot run. Name it musetalk and let its loader explain the
        # missing masks rather than silently trying wav2lip.
        return "musetalk", "found latents.pt but no mask/ directory"

    if _has_images(os.path.join(path, "face_imgs")):
        return "wav2lip", "found face_imgs/ crops and no latents.pt"

    # Legacy MuseTalk artifacts predate the mask directory but still carry the
    # info file written by its generator.
    if os.path.exists(os.path.join(path, "avator_info.json")):
        return "musetalk", "found avator_info.json (no latents.pt yet)"

    return None, (f"no recognizable artifact layout in {path} "
                  f"(expected latents.pt+mask/ for musetalk, "
                  f"face_imgs/ for wav2lip, ultralight.pth, "
                  f"or ditto.json for ditto)")


def detect_backend_or_none(avatar_id: str,
                           root: str | None = None) -> Optional[str]:
    return detect_backend(avatar_id, root)[0]
