#!/usr/bin/env python3
"""Download the avatar-preprocessing enhancement weights into models/.

Pulls Real-ESRGAN (super-resolution) and GFPGAN (face restore) into the repo's
models/ dir, matching scripts/download_musetalk_weights.py. Idempotent: skips
files that already exist.

Usage:
    /opt/miniconda3/envs/livetalking/bin/python scripts/download_enhance_weights.py
"""
import os
import sys
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS = os.path.abspath(os.path.join(ROOT, "models"))

WEIGHTS = {
    "RealESRGAN_x4plus.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "GFPGANv1.4.pth":
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
}


def main() -> int:
    os.makedirs(MODELS, exist_ok=True)
    for name, url in WEIGHTS.items():
        dest = os.path.join(MODELS, name)
        if os.path.isfile(dest):
            print(f"skip (exists): {name}")
            continue
        print(f"downloading {name} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"error downloading {name}: {exc}")
            return 1
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  ok ({size_mb:.1f} MB)")
    print("done. models present in:", MODELS)
    return 0


if __name__ == "__main__":
    sys.exit(main())