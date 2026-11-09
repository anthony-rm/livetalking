#!/usr/bin/env python3
"""CLI: enhance a raw talking-head video into a clean, vertical base video.

Usage:
    python -m tools.enhance.cli -i raw.mov -o clean.mp4
    python -m tools.enhance.cli -i raw.mov -o clean.mp4 \
        --width 1080 --height 1920 --background '#1E1E1E' --matte-out matte/

Stage status (see tools/enhance/pipeline.py):
    frame read + H.264 encode  -> implemented
    enhance_frame (quality)    -> implemented (Task 4: lighting + Real-ESRGAN + GFPGAN)
    compute_alpha (matting)    -> implemented (Task 3, RVM via torch.hub)
    compose_vertical           -> implemented (flat background)
The tool removes the background, enhances the person (lighting + face restore,
optionally super-res), composites onto a flat-color 9:16 canvas, and can write a
matte sidecar.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterator

import numpy as np

from tools.enhance import pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="enhance",
        description="Preprocess a talking-head video into a clean vertical base video for LiveTalking.",
    )
    p.add_argument("-i", "--input", required=True, help="input video path")
    p.add_argument("-o", "--output", required=True, help="output mp4 path")
    p.add_argument("--width", type=int, default=pipeline.DEFAULT_WIDTH,
                   help=f"canvas width (default {pipeline.DEFAULT_WIDTH})")
    p.add_argument("--height", type=int, default=pipeline.DEFAULT_HEIGHT,
                   help=f"canvas height (default {pipeline.DEFAULT_HEIGHT})")
    p.add_argument("--background", default=pipeline.DEFAULT_BG,
                   help="flat background color: '#RRGGBB', '#RGB' or 'r,g,b' (default %(default)s)")
    p.add_argument("--fit", choices=["contain", "cover"], default="contain",
                   help="contain = letterbox to fit; cover = fill and center-crop (default contain)")
    p.add_argument("--strength", type=float, default=0.5,
                   help="enhancement strength 0..1 (blend toward full correction; default 0.5)")
    p.add_argument("--upscale", type=int, default=1,
                   help="Real-ESRGAN upscale factor (default 1 = face-restore at "
                        "native res, fast; set 2/4 for genuinely low-res sources, "
                        "much slower)")
    p.add_argument("--no-face-restore", action="store_true",
                   help="skip GFPGAN face restoration (SR/lighting only)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="process at most N frames (fast smoke tests)")
    p.add_argument("--matte-out", default=None,
                   help="write person-matte sidecar here (grayscale PNGs + meta.json)")
    p.add_argument("--person-out", default=None,
                   help="write the person color layer sidecar here (BGR PNGs, same "
                        "geometry as the matte — needed for a lossless background swap)")
    p.add_argument("--recompose-from", default=None,
                   help="re-compose an existing --matte-out dir onto a new "
                        "--background (no re-matting, no re-enhancing)")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    return p


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        pipeline.parse_color(args.background)
    except ValueError as e:
        _log(f"error: {e}")
        return 2

    if not args.quiet:
        _log(f"input : {args.input}")
        _log(f"output: {args.output}  ({args.width}x{args.height}, fit={args.fit})")

    if args.recompose_from:
        import cv2
        cap, fps = pipeline.open_reader(args.input)  # fps source only
        cap.release()
        result = pipeline.recompose_video(
            args.recompose_from, args.output,
            person_dir=args.person_out, background=args.background, fps=fps)
        if not args.quiet:
            _log(f"recomposed {result['frames']} frames from "
                 f"{args.recompose_from} -> {args.output}")
        return 0

    def progress(pct: int, phase: str) -> None:
        if not args.quiet and pct % 10 == 0:
            _log(f"  [{phase}] {pct}%")

    result = pipeline.enhance_video(
        args.input, args.output,
        width=args.width, height=args.height, background=args.background,
        fit=args.fit, strength=args.strength,
        face_restore=not args.no_face_restore, upscale=args.upscale,
        max_frames=args.max_frames, matte_out=args.matte_out,
        person_out=args.person_out, progress_callback=progress)

    if not args.quiet:
        _log(f"done: {result['frames']} frames -> {args.output}")
        if args.matte_out:
            _log(f"matte sidecar: {args.matte_out} (+ meta.json)"
                 + (f", person layer: {args.person_out}" if args.person_out else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
