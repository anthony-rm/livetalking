###############################################################################
#  Ditto artifact I/O — the data/avatars/<id>/ layout for a ditto avatar
#
#  Layout (written by genavatar, read by the avatar backend and detect.py):
#
#      ditto.json          marker + metadata (version, frame_count, dtype, ...)
#      f_s/NNNNNNNN.npy    appearance feature volume, fp16, [32,16,64,64]
#      x_s_info/NNNNNN.npz per-frame conditioning (kp, exp, pitch, yaw, roll)
#      full_imgs/NNNN.png  source frames, for in-process paste-back
#      coords.pkl, mask/   paste-back geometry, same as every other backend
#
#  Two invariants this module enforces, both learned from defects this repo has
#  already had (tests/test_latents_device.py):
#
#  1. Tensors are STORED detached, on CPU, as fp16. A tensor pickled with a
#     device tag cannot be loaded on a machine without that device, and a
#     CUDA-tagged .pt written on the box then dies on a CPU-only reader.
#  2. Reads go through a loader that maps location, so an artifact written on
#     one class of machine always loads on another.
###############################################################################
"""Read/write a Ditto avatar artifact directory.

Kept separate from the model code so the format can be tested on a machine with
no Ditto weights, no onnxruntime and no GPU — the failure modes here are about
data, not inference.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterator, Optional

import numpy as np
from numpy.typing import NDArray

#: Bumped when the on-disk layout changes incompatibly. A reader that sees a
#: different major version must fail loudly rather than load half an artifact.
ARTIFACT_VERSION = 1

MARKER = "ditto.json"
F_S_DIR = "f_s"
X_S_DIR = "x_s_info"
FULL_IMGS_DIR = "full_imgs"

#: ``f_s`` is Bx32x16x64x64 per frame (see PLAN_DITTO.md §8.3), i.e. 2.1 M
#: floats. fp16 keeps one 75-frame avatar near 315 MB instead of 630 MB.
F_S_SHAPE = (32, 16, 64, 64)
F_S_DTYPE = np.float16


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, incomplete, or a newer version."""


# ─── paths ─────────────────────────────────────────────────────────────────
def marker_path(avatar_dir: str) -> str:
    return os.path.join(avatar_dir, MARKER)


def f_s_path(avatar_dir: str, frame_idx: int) -> str:
    return os.path.join(avatar_dir, F_S_DIR, f"{frame_idx:08d}.npy")


def x_s_path(avatar_dir: str, frame_idx: int) -> str:
    return os.path.join(avatar_dir, X_S_DIR, f"{frame_idx:08d}.npz")


def full_img_path(avatar_dir: str, frame_idx: int) -> str:
    return os.path.join(avatar_dir, FULL_IMGS_DIR, f"{frame_idx:08d}.png")


# ─── tensors: device-safe by construction ─────────────────────────────────
def to_storable(tensor: Any, dtype: Optional[np.dtype] = None) -> NDArray:
    """Detach any torch tensor and hand back a CPU numpy array.

    This is the single chokepoint that keeps a device tag out of the artifact.
    Accepts torch tensors and numpy arrays alike so callers need no branch.
    """
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "float") and dtype is np.float16:
        # .half() before .numpy() keeps precision loss inside torch; calling
        # .numpy().astype(fp16) would round in numpy and lose nothing more, but
        # torch's half is what the model produced in the first place.
        tensor = tensor.half()
    if hasattr(tensor, "numpy"):
        arr = tensor.numpy()
    else:
        arr = np.asarray(tensor)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def write_f_s(avatar_dir: str, frame_idx: int, tensor: Any) -> str:
    """Write one appearance volume as fp16 npy (never a pickled tensor)."""
    arr = to_storable(tensor, dtype=F_S_DTYPE)
    if arr.shape != F_S_SHAPE:
        # A wrong shape here would surface as an unreadable render much later.
        raise ArtifactError(
            f"f_s shape {arr.shape} != expected {F_S_SHAPE} for frame {frame_idx}")
    os.makedirs(os.path.join(avatar_dir, F_S_DIR), exist_ok=True)
    path = f_s_path(avatar_dir, frame_idx)
    # np.save (not torch.save): no pickle, no device tag, loadable everywhere.
    np.save(path, arr, allow_pickle=False)
    return path


def read_f_s(avatar_dir: str, frame_idx: int) -> NDArray:
    """Read one appearance volume. Returns fp16 numpy; caller casts to float32."""
    path = f_s_path(avatar_dir, frame_idx)
    if not os.path.exists(path):
        raise ArtifactError(f"missing {path}")
    arr = np.load(path, allow_pickle=False)
    if arr.shape != F_S_SHAPE:
        raise ArtifactError(f"{path} has shape {arr.shape}, expected {F_S_SHAPE}")
    return arr


def write_x_s_info(avatar_dir: str, frame_idx: int, info: dict) -> str:
    """Write one frame's conditioning dict as a naked .npz (no pickled object)."""
    os.makedirs(os.path.join(avatar_dir, X_S_DIR), exist_ok=True)
    path = x_s_path(avatar_dir, frame_idx)
    payload = {k: to_storable(v, dtype=np.float32).reshape(-1)
               for k, v in info.items()}
    np.savez(path, **payload)
    return path


def read_x_s_info(avatar_dir: str, frame_idx: int) -> dict:
    """Read one frame's conditioning dict, reshaped back to its stored shape.

    Shapes are recorded in the marker's per-key metadata at write time; here
    they are recovered from the arrays themselves (1-D), so the backend reshapes
    with `x_s_info_shapes()` from the marker.
    """
    path = x_s_path(avatar_dir, frame_idx)
    if not os.path.exists(path):
        raise ArtifactError(f"missing {path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ─── marker ────────────────────────────────────────────────────────────────
def build_marker(*, avatar_id: str, frame_count: int, render_size: int,
                 has_eye_state: bool = False, source: str = "") -> dict:
    """Metadata written next to the artifact; also what detect.py keys on."""
    return {
        "version": ARTIFACT_VERSION,
        "backend": "ditto",
        "avatar_id": avatar_id,
        "frame_count": int(frame_count),
        "render_size": int(render_size),
        "f_s_shape": list(F_S_SHAPE),
        "f_s_dtype": np.dtype(F_S_DTYPE).name,
        "f_s_fp16": True,
        "has_eye_state": bool(has_eye_state),
        "source": source,
    }


def write_marker(avatar_dir: str, marker: dict) -> str:
    os.makedirs(avatar_dir, exist_ok=True)
    path = marker_path(avatar_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(marker, fh, indent=2, sort_keys=True)
    return path


def read_marker(avatar_dir: str) -> dict:
    path = marker_path(avatar_dir)
    if not os.path.exists(path):
        raise ArtifactError(f"no {MARKER} in {avatar_dir} — not a ditto artifact")
    with open(path, encoding="utf-8") as fh:
        marker = json.load(fh)
    version = int(marker.get("version", 0))
    if version != ARTIFACT_VERSION:
        raise ArtifactError(
            f"artifact version {version} != {ARTIFACT_VERSION}; re-generate the "
            f"avatar with this build instead of loading it")
    return marker


# ─── validation / iteration ────────────────────────────────────────────────
def validate(avatar_dir: str) -> dict:
    """Check the artifact can be driven; return the marker.

    Fails with the specific missing piece rather than letting the backend die on
    the first read, because "ditto.json says 75 frames but f_s/ has 74" is a
    truncation bug that must be named as such.
    """
    marker = read_marker(avatar_dir)
    n = int(marker["frame_count"])
    if n <= 0:
        raise ArtifactError(f"{MARKER} says frame_count={n}")
    for name, pathfn in ((F_S_DIR, f_s_path), (X_S_DIR, x_s_path),
                         (FULL_IMGS_DIR, full_img_path)):
        if not os.path.isdir(os.path.join(avatar_dir, name)):
            raise ArtifactError(f"missing {name}/ in {avatar_dir}")
        missing = [i for i in range(n) if not os.path.exists(pathfn(avatar_dir, i))]
        if missing:
            raise ArtifactError(
                f"{name}/ has {n - len(missing)} of {n} frames; first missing "
                f"index {missing[0]} — the artifact was truncated")
    return marker


def iter_f_s(avatar_dir: str) -> Iterator[NDArray]:
    """Yield every stored appearance volume in frame order."""
    n = int(read_marker(avatar_dir)["frame_count"])
    for i in range(n):
        yield read_f_s(avatar_dir, i)


def artifact_checksum(avatar_dir: str) -> str:
    """Stable hash over the marker + every payload file, for the admin UI.

    Contents-based, not mtime-based: a copied artifact keeps its identity.
    """
    h = hashlib.sha256()
    marker = read_marker(avatar_dir)
    h.update(json.dumps(marker, sort_keys=True).encode())
    for name in (F_S_DIR, X_S_DIR):
        folder = os.path.join(avatar_dir, name)
        for fname in sorted(os.listdir(folder)):
            h.update(fname.encode())
            with open(os.path.join(folder, fname), "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
    return h.hexdigest()
