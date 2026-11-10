"""Dependency gate for the remote GPU worker (runs on the box).

Why this is a separate module
-----------------------------
``remote.worker`` imports ``grpc``, ``numpy`` and the generated protobuf stubs at
MODULE SCOPE. A ``--check`` flag inside it therefore can never report a missing
dependency: the interpreter fails while importing the checker, and the user gets
a traceback naming a file path instead of the module they need to install. This
module imports nothing heavy at module level and checks each dependency
individually, so a broken environment produces an actionable list.

Usage (on the box, from the repo root)::

    python -m remote.preflight          # transport deps + device report
    python -m remote.preflight --deep   # also the avatar-backend deps
    python -m remote.preflight --json   # machine-readable, for the admin UI

Exit codes: 0 all good, 2 a required module is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# What the worker needs merely to START (bind the port and serve Health).
# Backends and tools.enhance are imported lazily inside RPC handlers, so a
# minimal install gets the port listening — the right first step on a fresh box.
REQUIRED = [
    ("grpc", "gRPC transport"),
    ("google.protobuf", "generated inference_pb2 imports it; grpcio does NOT bundle it"),
    ("numpy", "tensors on the wire"),
    ("cv2", "image handling in utils/image.py"),
    ("tqdm", "progress bars in the pipelines"),
    ("torch", "the whole GPU path"),
]

# Needed by the avatar backends the worker actually runs. Checked with --deep
# because they are background-imported only when a model/avatar is requested.
BACKEND_DEPS = [
    ("einops", "musetalk UNet infer (musetalk/utils/utils.py)"),
    ("diffusers", "AutoencoderKL VAE load"),
    ("transformers", "whisper audio feature model"),
    ("omegaconf", "musetalk config loading"),
    ("soundfile", "audio IO"),
    ("resampy", "resampling"),
    ("scipy", "audio + numeric helpers"),
    ("av", "PyAV frames (video read)"),
    ("librosa", "wav2lip mel features"),
]

# Avatar CREATION on the box (GenerateAvatar) additionally needs landmarks.
GENAVATAR_DEPS = [
    ("face_recognition", "musetalk genavatar landmark + bbox detection"),
    ("dlib", "face_recognition backend (dlib-bin wheel)"),
]

# The base-video enhance pipeline.
ENHANCE_DEPS = [
    ("gfpgan", "face restore"),
    ("realesrgan", "super resolution"),
    ("basicsr", "pulled in transitively by the two above"),
    ("facexlib", "GFPGAN face detection/parsing"),
]


def _probe(modules):
    """Return [(module, purpose, error|None)] — never raises."""
    out = []
    for mod, purpose in modules:
        try:
            __import__(mod)
            out.append((mod, purpose, None))
        except Exception as exc:  # noqa: BLE001 - report, never propagate
            out.append((mod, purpose, f"{type(exc).__name__}: {exc}"))
    return out


def _device_report():
    info = {"python": sys.executable, "cwd": os.getcwd()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / 1e9, 1)
            info["capability"] = f"{props.major}.{props.minor}"
        try:
            from remote.revision import current_revision
            info["revision"] = current_revision()
        except Exception:  # noqa: BLE001
            info["revision"] = "unknown"
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def run(deep: bool = False) -> dict:
    checks = {"required": _probe(REQUIRED)}
    if deep:
        checks["backend"] = _probe(BACKEND_DEPS)
        checks["genavatar"] = _probe(GENAVATAR_DEPS)
        checks["enhance"] = _probe(ENHANCE_DEPS)

    missing_required = [c for c in checks["required"] if c[2]]
    report = {
        "ok": not missing_required,
        "device": _device_report(),
        "checks": {
            group: [{"module": m, "purpose": p, "error": e} for m, p, e in rows]
            for group, rows in checks.items()
        },
        "missing": [m for m, _p, _e in missing_required],
    }
    if deep:
        # A missing backend/genavatar/enhance dep does not stop the worker from
        # binding its port, but it DOES stop the feature that needs it — so
        # report them without failing the transport gate.
        report["missing_features"] = {
            group: [m for m, _p, _e in rows if _e]
            for group, rows in checks.items() if group != "required"
        }
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Remote GPU worker dependency gate")
    p.add_argument("--deep", action="store_true",
                   help="also check the backend, genavatar and enhance imports")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    report = run(deep=args.deep)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 2

    dev = report["device"]
    print(f"python   {dev.get('python')}")
    print(f"cwd      {dev.get('cwd')}")
    print(f"revision {dev.get('revision', 'unknown')}")
    if "torch_error" in dev:
        print(f"torch    ERROR {dev['torch_error']}")
    else:
        gpu = f" ({dev.get('gpu')}, {dev.get('vram_gb')} GB)" if dev.get("cuda") else ""
        print(f"torch    {dev.get('torch')}  cuda={dev.get('cuda')}{gpu}")

    for group, rows in report["checks"].items():
        bad = [r for r in rows if r["error"]]
        status = "OK" if not bad else f"{len(bad)} MISSING"
        print(f"\n[{group}] {status}")
        for r in rows:
            if r["error"]:
                print(f"  MISS {r['module']:<18} {r['purpose']}")
                print(f"       {r['error']}")
            elif args.deep or group == "required":
                print(f"  OK   {r['module']}")

    if not report["ok"]:
        missing = " ".join(report["missing"])
        print(f"\nmissing required: {missing}", file=sys.stderr)
        print("install with:  python -m pip install " + missing, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
