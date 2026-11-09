#!/usr/bin/env python3
"""Download MuseTalk model weights into ./models/.

Replicates the official MuseTalk download_weights.sh, but uses the Python
huggingface_hub API instead of the `hf` CLI (the CLI is deprecated and the
`hf-cli` skill is not installed in this env). Downloads are resumable.

Required for MuseTalk avatar generation (avatars/musetalk/utils/utils.py
loads the VAE from models/sd-vae; genavatar.py also needs musetalkV15,
face-parse-bisent and whisper).

Usage:
    python scripts/download_musetalk_weights.py [v1 | v15 | all]
"""
import os
import sys
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

# repo -> [(local_subdir, path_inside_repo)]
WANT = {
    "v1":  [("musetalk", "musetalk/musetalk.json"), ("musetalk", "musetalk/pytorch_model.bin")],
    "v15": [("musetalkV15", "musetalkV15/musetalk.json"), ("musetalkV15", "musetalkV15/unet.pth")],
    "vae": [("sd-vae", "config.json"), ("sd-vae", "diffusion_pytorch_model.bin")],
    "whisper": [("whisper", "config.json"), ("whisper", "pytorch_model.bin"),
                ("whisper", "preprocessor_config.json")],
    "syncnet": [("syncnet", "latentsync_syncnet.pt")],
    "dwpose": [("dwpose", "dw-ll_ucoco_384.pth")],
}


def fetch(repo, subdir, repo_path):
    dest_dir = os.path.join(MODELS, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    local_name = os.path.basename(repo_path)
    path = os.path.join(dest_dir, local_name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  skip (exists) {os.path.relpath(path, ROOT)}")
        return
    print(f"  -> {repo}/{repo_path}")
    # hf_hub_download has two constraints that fight each other:
    #   * `filename` must be the full path inside the repo, or the lookup 404s
    #     (passing just "musetalk.json" resolves against the repo root).
    #   * it preserves the leading directory of `filename` inside `local_dir`,
    #     so local_dir must be the PARENT of where you want the file.
    # For nested files (musetalkV15/unet.pth) that means local_dir=MODELS.
    # For root-level files (config.json) it means local_dir=dest_dir.
    local_dir = MODELS if "/" in repo_path else dest_dir
    hf_hub_download(repo, repo_path, local_dir=local_dir,
                    local_dir_use_symlinks=False)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        targets = ["v1", "v15", "vae", "whisper", "syncnet", "dwpose"]
    else:
        targets = [which]

    for t in targets:
        print(f"[{t}]")
        if t == "vae":
            fetch("stabilityai/sd-vae-ft-mse", "sd-vae", "config.json")
            fetch("stabilityai/sd-vae-ft-mse", "sd-vae", "diffusion_pytorch_model.bin")
        elif t == "whisper":
            fetch("openai/whisper-tiny", "whisper", "config.json")
            fetch("openai/whisper-tiny", "whisper", "pytorch_model.bin")
            fetch("openai/whisper-tiny", "whisper", "preprocessor_config.json")
        elif t == "syncnet":
            fetch("ByteDance/LatentSync", "syncnet", "latentsync_syncnet.pt")
        elif t == "dwpose":
            fetch("yzd-v/DWPose", "dwpose", "dw-ll_ucoco_384.pth")
        else:
            for sub, f in WANT[t]:
                fetch("TMElyralab/MuseTalk", sub, f)

    # Face-parse-bisent: 79999_iter.pth is a Google Drive file, not on HF.
    fp_dir = os.path.join(MODELS, "face-parse-bisent")
    os.makedirs(fp_dir, exist_ok=True)
    iter_path = os.path.join(fp_dir, "79999_iter.pth")
    if not os.path.exists(iter_path) or os.path.getsize(iter_path) == 0:
        print("[face-parse-bisent] 79999_iter.pth via gdown (Google Drive)")
        import gdown
        gdown.download(id="154JgKpzCPW82qINcVieuPH3fZ2e0P812",
                       output=iter_path, quiet=False)
    resnet_path = os.path.join(fp_dir, "resnet18-5c106cde.pth")
    if not os.path.exists(resnet_path) or os.path.getsize(resnet_path) == 0:
        print("[face-parse-bisent] resnet18-5c106cde.pth via torch CDN")
        import urllib.request
        urllib.request.urlretrieve(
            "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            resnet_path)

    print("\ndone.")


if __name__ == "__main__":
    main()