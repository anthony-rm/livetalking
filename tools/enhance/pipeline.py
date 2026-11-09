"""Avatar base-video preprocessing pipeline.

Stage seams (filled in by later tasks):
  enhance_frame()  -> Task 4 (lighting + Real-ESRGAN + GFPGAN)
  compute_alpha()  -> Task 3 (RVM person matting)
  compose_vertical() and the frame I/O below are real, reusable code.

The CLI (tools/enhance/cli.py) is a thin wrapper over this module.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920
DEFAULT_BG = "#1E1E1E"


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def parse_color(value: str) -> Tuple[int, int, int]:
    """Parse '#RRGGBB', '#RGB' or 'r,g,b' into an OpenCV BGR tuple."""
    v = value.strip()
    if v.startswith("#"):
        h = v[1:]
        if len(h) == 3:
            r, g, b = (int(c * 2, 16) for c in h)
        elif len(h) == 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        else:
            raise ValueError(f"bad hex color: {value!r}")
        return (b, g, r)
    parts = [int(float(x)) for x in v.split(",")]
    if len(parts) != 3:
        raise ValueError(f"bad rgb color: {value!r} (want 'r,g,b')")
    r, g, b = parts
    return (b, g, r)


# --------------------------------------------------------------------------- #
# Frame I/O
# --------------------------------------------------------------------------- #
def open_reader(path: str) -> Tuple[cv2.VideoCapture, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps != fps:  # nan / zero guard
        fps = 30.0
    return cap, fps


def iter_frames(cap: cv2.VideoCapture, max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        n += 1
        if max_frames is not None and n >= max_frames:
            break


def encode_video(frames: Iterator[np.ndarray], out_path: str, fps: float,
                 size: Tuple[int, int]) -> int:
    """Encode BGR frames to an H.264 MP4. Returns the number of frames written.

    Prefers ffmpeg (guarantees a broadly playable H.264/yuv420p file that the
    LiveTalking avatar generators can ingest); falls back to cv2.VideoWriter.
    """
    width, height = size
    ff = shutil.which("ffmpeg")
    count = 0
    if ff:
        cmd = [
            ff, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", out_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        try:
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
                count += 1
            proc.stdin.close()
        finally:
            ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg encode failed (exit {ret})")
        return count

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not vw.isOpened():
        raise RuntimeError("cv2.VideoWriter could not be opened and ffmpeg is unavailable")
    try:
        for frame in frames:
            vw.write(frame)
            count += 1
    finally:
        vw.release()
    return count


# --------------------------------------------------------------------------- #
# Stage seams (stubs for now)
# --------------------------------------------------------------------------- #
def enhance_frame(frame_bgr: np.ndarray, strength: float = 0.5,
                  face_restore: bool = True, upscale: int = 4) -> tuple:
    """Task 4: lighting + Real-ESRGAN + GFPGAN.

    Returns (enhanced_bgr, scale). scale lets the caller resize the person matte
    so it stays aligned with the (possibly upscaled) enhanced frame.
    """
    from tools.enhance.enhance import get_default_enhancer
    enhancer = get_default_enhancer(upscale=upscale, face_restore=face_restore)
    return enhancer.enhance(frame_bgr, strength)


def reset_enhancer() -> None:
    """Drop the cached default Enhancer (frees model memory)."""
    from tools.enhance.enhance import reset_default_enhancer
    reset_default_enhancer()


def compute_alpha(frame_bgr: np.ndarray, runner=None) -> Optional[np.ndarray]:
    """Task 3: RVM person matte for a single frame (uint8 HxW, 255=person).

    RVM is recurrent, so the runner carries hidden state across frames; if none
    is passed the module-level default runner is used (created lazily).
    Returns None only when matting is disabled.
    """
    if runner is None:
        from tools.enhance.matting import get_default_runner
        runner = get_default_runner()
    return runner.alpha(frame_bgr)


def reset_matte_state() -> None:
    """Drop RVM recurrent state — call at the start of each new stream."""
    from tools.enhance.matting import reset_default_runner
    reset_default_runner()


def write_matte_sidecar(alpha: np.ndarray, out_dir: str, index: int) -> str:
    """Persist one alpha frame as a grayscale PNG (re-export from matting)."""
    from tools.enhance.matting import write_matte_sidecar as _write
    return _write(alpha, out_dir, index)


def load_matte_sidecar(matte_dir: str, index: int) -> np.ndarray:
    """Load one alpha frame from a `--matte-out` dir (grayscale PNG, 255=person)."""
    path = os.path.join(matte_dir, f"{index:08d}.png")
    a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(f"no matte sidecar at {path}")
    return a


def load_person_sidecar(person_dir: str, index: int) -> Optional[np.ndarray]:
    """Load one person color frame from a `--person-out` dir (BGR PNG)."""
    path = os.path.join(person_dir, f"{index:08d}.png")
    img = cv2.imread(path)
    return img


# --------------------------------------------------------------------------- #
# Whole-video jobs (used by the CLI and the web server)
# --------------------------------------------------------------------------- #
def _person_layer_fn(person_dir: Optional[str]):
    if person_dir is None:
        return None
    def layer(idx: int):
        return load_person_sidecar(person_dir, idx)
    return layer


def enhance_video(input_path: str, output_path: str, *,
                  width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
                  background: str = DEFAULT_BG, fit: str = "contain",
                  strength: float = 0.5, face_restore: bool = True,
                  upscale: int = 1, max_frames: Optional[int] = None,
                  matte_out: Optional[str] = None,
                  person_out: Optional[str] = None,
                  progress_callback=None, should_cancel=None) -> dict:
    """Run the full enhance pipeline on a video file.

    progress_callback(percent:int, phase:str) is called with the overall 0-100
    progress and a phase name ('matting', 'enhance', 'encode', 'done').
    should_cancel() -> bool lets a caller abort between frames.
    Returns a dict with frames/fps/geometry.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"input not found: {input_path}")
    bg_bgr = parse_color(background)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    for side in (matte_out, person_out):
        if side:
            os.makedirs(side, exist_ok=True)

    def report(pct: int, phase: str) -> None:
        if progress_callback:
            progress_callback(int(pct), phase)

    report(1, "starting")
    cap, fps = open_reader(input_path)
    reset_matte_state()
    reset_enhancer()

    def frames():
        import numpy as _np
        n = 0
        total = None
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        except Exception:
            total = None
        if max_frames:
            total = min(total, max_frames) if total else max_frames
        for frame in iter_frames(cap, max_frames):
            if should_cancel and should_cancel():
                raise Cancelled()
            alpha = compute_alpha(frame)
            if matte_out:
                write_matte_sidecar(alpha, matte_out, n)
            if person_out:
                cv2.imwrite(os.path.join(person_out, f"{n:08d}.png"), frame)
            frame, scale = enhance_frame(frame, strength,
                                         face_restore=face_restore,
                                         upscale=upscale)
            if scale != 1:
                alpha = cv2.resize(alpha, (frame.shape[1], frame.shape[0]),
                                   interpolation=cv2.INTER_AREA)
            out = compose_vertical(frame, alpha, width, height, bg_bgr, fit)
            n += 1
            if total:
                report(min(95, 2 + int(n / total * 90)), "enhance")
            else:
                report(min(95, 2 + n), "enhance")
            yield out

    try:
        written = encode_video(frames(), output_path, fps, (width, height))
    except Cancelled:
        if os.path.isfile(output_path):
            os.remove(output_path)
        raise
    finally:
        cap.release()

    if matte_out:
        import json
        meta = {
            "input": os.path.abspath(input_path),
            "output": os.path.abspath(output_path),
            "frames": written,
            "fps": fps,
            "canvas": [width, height],
            "fit": fit,
            "background": list(bg_bgr),
            "note": "alpha PNGs are at the SOURCE frame resolution, named "
                    "00000000.png..; compose with the same canvas/fit to reuse.",
        }
        with open(os.path.join(matte_out, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    report(100, "done")
    return {"frames": written, "fps": fps, "canvas": [width, height],
            "fit": fit, "output": os.path.abspath(output_path)}


def recompose_video(matte_dir: str, output_path: str, *,
                    person_dir: Optional[str] = None,
                    background: str = DEFAULT_BG,
                    fps: float = 30.0, progress_callback=None) -> dict:
    """Re-compose a saved matte/person sidecar onto a new background (fast)."""
    meta_path = os.path.join(matte_dir, "meta.json")
    import json
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    width, height = meta["canvas"]
    bg_bgr = parse_color(background)

    def frames():
        n = 0
        total = meta.get("frames") or 0
        while True:
            try:
                alpha = load_matte_sidecar(matte_dir, n)
            except FileNotFoundError:
                break
            person = load_person_sidecar(person_dir, n) if person_dir else None
            if person is None:
                person = np.zeros((alpha.shape[0], alpha.shape[1], 3), np.uint8)
            yield compose_vertical(person, alpha, width, height, bg_bgr, meta.get("fit", "contain"))
            n += 1
            if progress_callback and total:
                progress_callback(min(95, int(n / total * 95)), "recompose")

    written = encode_video(frames(), output_path, fps, (width, height))
    if progress_callback:
        progress_callback(100, "done")
    return {"frames": written, "fps": fps, "canvas": [width, height],
            "output": os.path.abspath(output_path)}


class Cancelled(RuntimeError):
    """Raised when a job's should_cancel() returns True."""
    pass


def recompose_from_matte(matte_dir: str, bg_bgr: Tuple[int, int, int],
                         width: int, height: int, fps: float,
                         fit: str = "contain", out_path: Optional[str] = None,
                         person_layer=None) -> Iterator[np.ndarray]:
    """Re-compose a saved person-matte onto a new background.

    This is the single seam a later in-app background picker calls: it takes the
    matte sidecar (source-resolution PNGs) and a background color/image and
    yields opaque vertical frames — no re-matting, no re-enhancing.

    `person_layer(index)` returns the person RGB frame for index (same
    resolution/geometry as the matte). If None, the seam yields the background
    with the person silhouette cut out (a transparent-person result) — useful
    for testing, but a real swap needs the person color, which the CLI saves as
    a `--person-out` sidecar.
    """
    import json
    meta_path = os.path.join(matte_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("fit") != fit:
        raise ValueError(f"matte was produced with fit={meta.get('fit')!r}, "
                         f"got {fit!r}")
    if meta.get("canvas") != [width, height]:
        raise ValueError(f"matte canvas {meta.get('canvas')} != {[width, height]}")

    n = 0
    while True:
        try:
            alpha = load_matte_sidecar(matte_dir, n)
        except FileNotFoundError:
            break
        person = person_layer(n) if person_layer is not None else None
        if person is None:
            person = np.zeros((alpha.shape[0], alpha.shape[1], 3), np.uint8)
        out = compose_vertical(person, alpha, width, height, bg_bgr, fit)
        yield out
        n += 1


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def _scale_fit(frame: np.ndarray, width: int, height: int,
               fit: str) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    scale = (max if fit == "cover" else min)(width / w, height / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    resized = cv2.resize(frame, (nw, nh), interpolation=interp)

    if fit == "cover":
        x0 = (nw - width) // 2
        y0 = (nh - height) // 2
        crop = resized[y0:y0 + height, x0:x0 + width]
        return crop, (0, 0, width, height)

    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    return resized, (x0, y0, x0 + nw, y0 + nh)


def compose_vertical(frame_bgr: np.ndarray, alpha: Optional[np.ndarray],
                     width: int, height: int, bg_bgr: Tuple[int, int, int],
                     fit: str = "contain") -> np.ndarray:
    """Place the person layer onto a flat-color portrait canvas.

    When `alpha` is provided it must share the frame's HxW; it is resized with
    the same geometry so the matte stays aligned with the person.
    """
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = bg_bgr

    if alpha is None:
        fg, (x0, y0, x1, y1) = _scale_fit(frame_bgr, width, height, fit)
        canvas[y0:y1, x0:x1] = fg
        return canvas

    fg_c, (x0, y0, x1, y1) = _scale_fit(frame_bgr, width, height, fit)
    a_c, _ = _scale_fit(alpha, width, height, fit)
    if a_c.ndim == 3:
        a_c = a_c[:, :, 0]
    region = canvas[y0:y1, x0:x1]
    a = (a_c.astype(np.float32) / 255.0)[:, :, None]
    region[:] = (fg_c.astype(np.float32) * a
                 + region.astype(np.float32) * (1.0 - a)).astype(np.uint8)
    return canvas
