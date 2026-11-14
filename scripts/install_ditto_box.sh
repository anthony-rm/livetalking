#!/usr/bin/env bash
# Install Ditto STANDALONE on a fresh GPU box (Phase 0 of PLAN_DITTO.md).
#
# This does NOT touch the LiveTalking app. It creates one conda env, installs
# Ditto's Python deps, vendors the upstream source at a pinned commit, fetches
# the PyTorch weight lane (2.31 GB), and runs upstream's example inference.
#
#   bash scripts/install_ditto_box.sh              # full install + smoke test
#   bash scripts/install_ditto_box.sh --env-only   # just create the env + deps
#   bash scripts/install_ditto_box.sh --weights    # just fetch weights + run
#   bash scripts/install_ditto_box.sh --dry-run    # print the plan, change nothing
#
# Why this script exists rather than a doc: every pin below is load-bearing and
# was resolved with pip against linux wheels, not guessed. See PLAN_DITTO.md §2.
#
# KEY CONSTRAINTS (do not "simplify" these away):
#   * mediapipe 0.10.21 requires numpy<2, so numpy is pinned to 1.26.4 for the
#     WHOLE env. Ditto's LMDM and the app both work on 1.26.
#   * onnxruntime-gpu must be >=1.20 for cu12 and have a cp312 wheel; 1.23.2 does.
#   * upstream needs python 3.10-3.12. We pick 3.12 to match the app env, and the
#     full dep set was verified to resolve on cp312.
#   * torch is installed from the cu124 index because the box driver is 560.94.
#     A bare `pip install torch` gets the CPU wheel and everything after is wasted.
#   * the upstream Cython blend extension is NOT built (needs gcc + distutils,
#     which py3.12 removed). Phase 0's upstream smoke test does need gcc; the
#     integrated backend never imports that module (PLAN_DITTO.md §5).
set -uo pipefail

UPSTREAM_REPO="https://github.com/antgroup/ditto-talkinghead"
UPSTREAM_COMMIT="c3e47eee2e626500017a0556b470d6d4182f85e8"
HF_REPO="digital-avatar/ditto-talkinghead"
ENV_NAME="livetalking"
SRC_ROOT="${DITTO_SRC_ROOT:-$HOME/ditto-probe}"
WEIGHT_ROOT="${DITTO_WEIGHT_ROOT:-$HOME/ditto-weights}"

DO_ENV=1
DO_WEIGHTS=1
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --env-only)   DO_WEIGHTS=0 ;;
    --weights)    DO_ENV=0 ;;
    --dry-run)    DRY_RUN=1 ;;
    -h|--help)    sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
run()  { if [ "$DRY_RUN" = 1 ]; then printf '   [dry-run] %s\n' "$*"; else "$@"; fi; }

if [ "$DRY_RUN" = 1 ]; then
  # A dry run must be runnable anywhere (a laptop reviewing the plan), so the
  # GPU/conda/disk checks are reported, never enforced. The real run enforces.
  GPUSIM=1
fi

# ─── 0. Preflight (read-only; the failures here are cheap, later ones are not) ─
say "0. Preflight"

# WSL2 puts nvidia-smi under /usr/lib/wsl/lib and does NOT add it to PATH for
# non-interactive shells. Bare-metal boxes have it in /usr/bin. Find it either way
# — a false "not a GPU box" here would abort a perfectly good install.
SMI=""
for c in /usr/bin/nvidia-smi /usr/local/bin/nvidia-smi /usr/lib/wsl/lib/nvidia-smi; do
  [ -x "$c" ] && { SMI="$c"; break; }
done
[ -n "$SMI" ] || SMI=$(command -v nvidia-smi 2>/dev/null || true)

if [ -z "$SMI" ]; then
  [ "$DRY_RUN" = 1 ] || die "nvidia-smi not found (looked in /usr/bin, /usr/local/bin, /usr/lib/wsl/lib): this is not the GPU box."
  note "nvidia-smi: MISSING (fine for --dry-run; the real run requires the GPU box)"
  GPU_LINE="(simulated)"
else
  note "nvidia-smi: $SMI"
  GPU_LINE=$("$SMI" --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)
fi
note "GPU: $GPU_LINE"

# On WSL2 the driver libs live in /usr/lib/wsl/lib. Without this, torch imports
# fine and then reports cuda.is_available() == False, which looks like a broken
# install rather than a missing LD_LIBRARY_PATH.
if [ -d /usr/lib/wsl/lib ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":/usr/lib/wsl/lib:"*) ;;
    *) export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
  note "WSL2 detected; LD_LIBRARY_PATH includes /usr/lib/wsl/lib"
fi

if [ -n "$SMI" ]; then
  DRIVER=$("$SMI" --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  [ "${DRIVER:-0}" -ge 550 ] 2>/dev/null \
    || note "WARNING: driver $DRIVER < 550; cu124 wheels need >=550. Expect a CUDA error later."
else
  note "driver: unknown (no nvidia-smi)"
fi
note "user=$(id -un) home=$HOME"
DF_AVAIL=$(df -Pk "$HOME" | awk 'NR==2{print int($4/1024/1024)}')
note "free space under \$HOME: ${DF_AVAIL} GB (need ~6 GB: 2.3 weights + ~2.5 env + source)"
[ "${DF_AVAIL:-0}" -ge 6 ] || die "not enough free space under \$HOME (need >=6 GB, have ${DF_AVAIL} GB)."

CONDA_BIN=""
for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
         "$HOME/.conda/miniconda3/bin/conda" "/opt/conda/bin/conda"; do
  [ -x "$c" ] && { CONDA_BIN="$c"; break; }
done
if [ -z "$CONDA_BIN" ] && command -v conda >/dev/null; then
  CONDA_BIN=$(command -v conda)
fi
if [ -z "$CONDA_BIN" ]; then
  [ "$DRY_RUN" = 1 ] || die "no conda found. Install miniconda under \$HOME first (no sudo):
  curl -fsSL -o /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/mc.sh -b -p \"\$HOME/miniconda3\""
  note "conda: MISSING (fine for --dry-run)"
  CONDA_BIN="conda"
fi
note "conda: $CONDA_BIN"
CONDA_PREFIX_BASE=$("$CONDA_BIN" info --base 2>/dev/null || echo "")
note "conda base: ${CONDA_PREFIX_BASE:-unknown}"

if command -v gcc >/dev/null; then note "gcc: $(gcc -dumpversion) (only needed by the upstream smoke test)"
else note "gcc: MISSING — the upstream smoke test may fail building the Cython blend ext.
     The integrated backend does not need it (PLAN_DITTO.md §5), but Phase 0 does.
     Workaround if it fails: apt is not available to a normal user, ask the admin."; fi

# ─── 1. Env ──────────────────────────────────────────────────────────────────
ENV_PY="$CONDA_PREFIX_BASE/envs/$ENV_NAME/bin/python"

if [ "$DO_ENV" = 1 ]; then
  say "1. conda env '$ENV_NAME' (python 3.12)"
  if [ -x "$ENV_PY" ]; then
    note "already exists: $ENV_PY"
  else
    run "$CONDA_BIN" create -y -n "$ENV_NAME" python=3.12
  fi
  if [ "$DRY_RUN" != 1 ]; then
    # Ask conda where the env actually landed instead of guessing between
    # <base>/envs and ~/.conda/envs — both layouts are common and the difference
    # silently breaks every later pip call.
    PREFIX=$("$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys;print(sys.prefix)' 2>/dev/null | tail -1)
    [ -n "$PREFIX" ] || PREFIX=$("$CONDA_BIN" env list 2>/dev/null | awk -v n="$ENV_NAME" '$1==n{print $NF}')
    if [ -n "$PREFIX" ] && [ -x "$PREFIX/bin/python" ]; then
      ENV_PY="$PREFIX/bin/python"
    elif [ -x "$HOME/.conda/envs/$ENV_NAME/bin/python" ]; then
      ENV_PY="$HOME/.conda/envs/$ENV_NAME/bin/python"
    fi
    [ -x "$ENV_PY" ] || die "could not locate the interpreter for env '$ENV_NAME'.
  Tried conda run/env list, <base>/envs, and ~/.conda/envs. Check: $CONDA_BIN env list"
    note "resolved prefix: $(dirname "$(dirname "$ENV_PY")")"
  fi
  note "python: $ENV_PY"

  say "1b. torch 2.5.1+cu124"
  if [ "$DRY_RUN" = 1 ]; then
    note "[dry-run] pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url .../cu124"
  else
    "$ENV_PY" -m pip install --upgrade pip -q
    "$ENV_PY" -m pip install \
      torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu124 -q \
      || die "torch install failed. Check the driver ($DRIVER) supports cu124."
    # This check is the whole point of pinning: a CPU wheel here wastes the box.
    "$ENV_PY" - <<'PY' || die "torch is not CUDA-enabled. Do NOT continue."
import sys, torch
print(f"   torch {torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit(1)
print(f"   gpu: {torch.cuda.get_device_name(0)}")
PY
  fi

  say "1c. Ditto dependencies (pinned; numpy<2 because of mediapipe)"
  DEPS=(
    "numpy==1.26.4"
    "mediapipe==0.10.21"
    "onnxruntime-gpu==1.23.2"
    "opencv-python-headless==4.10.0.84"
    "librosa==0.10.2.post1"
    "scikit-image==0.25.0"
    "scipy==1.15.0"
    "imageio==2.36.1"
    "imageio-ffmpeg==0.5.1"
    "einops==0.8.1"
    "filetype==1.2.0"
    "tqdm==4.67.1"
    "Cython==3.0.11"
    "soundfile==0.12.1"
  )
  run "$ENV_PY" -m pip install "${DEPS[@]}" -q

  say "1d. Verify the imports the model actually does"
  if [ "$DRY_RUN" = 1 ]; then
    note "[dry-run] import probe"
  else
    "$ENV_PY" - <<'PY' || die "a Ditto dependency does not import; see the line above."
import importlib, sys
bad = []
for m in ("numpy","torch","cv2","onnxruntime","mediapipe","librosa",
          "skimage","scipy","imageio","einops","filetype","Cython"):
    try:
        mod = importlib.import_module(m)
        print(f"   OK   {m:12s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"   MISS {m:12s} {e}")
        bad.append(m)
import numpy
print(f"   numpy {numpy.__version__} (must be <2)")
if int(numpy.__version__.split('.')[0]) >= 2:
    bad.append("numpy>=2")
try:
    import onnxruntime as ort
    print(f"   onnxruntime providers: {ort.get_available_providers()}")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("   WARNING: onnxruntime has no CUDA provider — HuBERT and the aux "
              "landmark models will run on CPU (slow, and possibly too slow).")
except Exception:
    pass
sys.exit(1 if bad else 0)
PY
  fi
else
  say "1. env: skipped (--weights)"
  [ -x "$ENV_PY" ] || die "no env at $ENV_PY; run without --weights first."
fi

# ─── 2. Upstream source ──────────────────────────────────────────────────────
if [ "$DO_WEIGHTS" = 1 ]; then
  say "2. upstream source at ${UPSTREAM_COMMIT:0:8} -> $SRC_ROOT"
  # A pre-extracted tree (no .git) is accepted: on a network where github.com is
  # unreachable the source can only arrive as a tarball, and re-cloning would
  # fail there. Verify by presence of the entrypoint instead of by HEAD.
  if [ -d "$SRC_ROOT/.git" ]; then
    note "git checkout exists; pinning to the pinned commit"
    run git -C "$SRC_ROOT" fetch --quiet origin "$UPSTREAM_COMMIT"
    run git -C "$SRC_ROOT" checkout --quiet "$UPSTREAM_COMMIT"
  elif [ -f "$SRC_ROOT/inference.py" ]; then
    note "using the pre-extracted tree at $SRC_ROOT (no .git; cannot verify commit)"
  else
    run mkdir -p "$SRC_ROOT"
    run git clone --quiet "$UPSTREAM_REPO" "$SRC_ROOT"
    run git -C "$SRC_ROOT" checkout --quiet "$UPSTREAM_COMMIT"
  fi
  if [ "$DRY_RUN" = 1 ]; then
    note "[dry-run] source at $UPSTREAM_COMMIT"
  elif [ -d "$SRC_ROOT/.git" ]; then
    HEAD_NOW=$(git -C "$SRC_ROOT" rev-parse HEAD)
    [ "$HEAD_NOW" = "$UPSTREAM_COMMIT" ] \
      || die "source is at $HEAD_NOW, expected $UPSTREAM_COMMIT"
    note "HEAD ok: $HEAD_NOW"
  else
    [ -f "$SRC_ROOT/inference.py" ] || die "no inference.py under $SRC_ROOT"
    note "source ok (tarball): $(ls "$SRC_ROOT" | wc -l) entries"
  fi

  # ─── 3. Weights (PyTorch lane only) ──────────────────────────────────────
  say "3. weights -> $WEIGHT_ROOT (ditto_pytorch + ditto_cfg, 2.31 GB)"
  if [ "$DRY_RUN" = 1 ]; then
    note "[dry-run] snapshot_download $HF_REPO patterns ditto_pytorch/* ditto_cfg/*"
  else
    # huggingface.co may be unreachable; HF_ENDPOINT lets a mirror or a proxy
    # serve it without editing code. Respect a proxy if one is exported too.
    note "HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co (default)}"
    [ -n "${HTTPS_PROXY:-${https_proxy:-}}" ] && note "using proxy $([ -n "${HTTPS_PROXY:-}" ] && echo "$HTTPS_PROXY" || echo "$https_proxy")"
    "$ENV_PY" -m pip install -q "huggingface_hub[hf_transfer]" 2>/dev/null || true
    if ! HF_HUB_ENABLE_HF_TRANSFER=1 HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
         "$ENV_PY" - "$HF_REPO" "$WEIGHT_ROOT" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
p = snapshot_download(
    repo_id=repo, local_dir=dest,
    allow_patterns=["ditto_pytorch/**", "ditto_cfg/**", "LICENSE", "README.md"],
)
print(f"   downloaded to {p}")
PY
    then
      die "weight download failed. If huggingface.co is unreachable, set
  export HF_ENDPOINT=https://hf-mirror.com   (or export HTTPS_PROXY=http://host:port)
and re-run with --weights. Progress is resumable."
    fi
    du -sh "$WEIGHT_ROOT/ditto_pytorch" 2>/dev/null || true
  fi
fi

# ─── 4. Smoke test ───────────────────────────────────────────────────────────
say "4. upstream example inference"
# The cfg must match the WEIGHT LANE, not the latency we want. The three shipped
# configs embed model paths, and only the pytorch cfg names .pth files:
#   v0.4_hubert_cfg_pytorch.pkl  -> models/*.pth        (lane: ditto_pytorch)
#   v0.4_hubert_cfg_trt*.pkl     -> *.engine            (lane: ditto_trt_*)
# Pairing the TRT cfg with pytorch weights fails on a missing .engine. There is
# no "pytorch online" cfg upstream; the online settings are runtime kwargs, so
# step 5 synthesises one rather than looking for a file (see below).
CFG="$WEIGHT_ROOT/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
DATA="$WEIGHT_ROOT/ditto_pytorch"
AUDIO="$SRC_ROOT/example/audio.wav"
IMAGE="$SRC_ROOT/example/image.png"
OUT="$SRC_ROOT/tmp/result.mp4"

[ -f "$CFG" ] || [ "$DRY_RUN" = 1 ] || die "missing $CFG (weights incomplete; re-run --weights)"
if [ "$DRY_RUN" = 1 ]; then
  note "[dry-run] time $ENV_PY inference.py --data_root $DATA --cfg_pkl <pytorch cfg> ..."
  exit 0
fi

if [ ! -f "$CFG" ]; then
  die "missing $CFG"
fi

mkdir -p "$SRC_ROOT/tmp"
START=$(date +%s)
( cd "$SRC_ROOT" && "$ENV_PY" inference.py \
    --data_root "$DATA" \
    --cfg_pkl "$CFG" \
    --audio_path "$AUDIO" \
    --source_path "$IMAGE" \
    --output_path "$OUT" )
RC=$?
END=$(date +%s)

if [ $RC -ne 0 ]; then
  say "SMOKE TEST FAILED (exit $RC)"
  note "If the error is about blend_images / Cython / distutils, that is the known"
  note "upstream Cython path needing gcc (PLAN_DITTO.md §5). Phase 0 needs it;"
  note "the integrated backend will not."
  exit $RC
fi

WALL=$((END-START))
DUR=$("$ENV_PY" - "$OUT" <<'PY' 2>/dev/null || echo ""
import sys
try:
    import imageio.v2 as iio
    r = iio.get_reader(sys.argv[1]); print(r.get_meta_data().get("duration",""))
except Exception:
    pass
PY
)

say "SMOKE TEST OK"
note "output : $OUT ($(du -h "$OUT" 2>/dev/null | cut -f1))"
note "wall   : ${WALL} s"
[ -n "$DUR" ] && note "video  : ${DUR} s"

say "5. Throughput measurements (the Phase 0 gate)"
( cd "$SRC_ROOT" && "$ENV_PY" - "$DATA" "$CFG" <<'PY' )
import sys, time, statistics, numpy as np
data_root, cfg_pkl = sys.argv[1], sys.argv[2]

from core.atomic_components.cfg import parse_cfg
(cfg_reg, cfg_cond, lmdm_cfg, stitch_cfg, warp_cfg, dec_cfg,
 w2f_cfg, dkw) = parse_cfg(cfg_pkl, data_root)

print(f"  shipped cfg    : overlap_v2={dkw['overlap_v2']} "
      f"steps={dkw['sampling_timesteps']} online={dkw['online_mode']}")
print("  NOTE upstream ships no PYTORCH 'online' cfg. The low-latency settings")
print("       are runtime kwargs, so both variants are measured below.")

from core.atomic_components.audio2motion import Audio2Motion
import torch

def measure_pass(overlap_v2, steps, online):
    """Median LMDM pass time for one latency config, on a fresh model."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    a2m = Audio2Motion(lmdm_cfg)
    a2m.setup(dkw["ch_info"]["x_s_info_lst"][0],
              overlap_v2=overlap_v2, fix_kp_cond=dkw["fix_kp_cond"],
              fix_kp_cond_dim=dkw["fix_kp_cond_dim"],
              sampling_timesteps=steps, online_mode=online,
              v_min_max_for_clip=dkw["v_min_max_for_clip"],
              smo_k_d=dkw["smo_k_d"])
    aud = np.random.randn(1, a2m.seq_frames,
                          lmdm_cfg["audio_feat_dim"]).astype(np.float32)
    a2m(aud)                                    # warm up allocator/cudnn
    ts = []
    for _ in range(10):
        t0 = time.perf_counter(); a2m(aud); ts.append((time.perf_counter()-t0)*1e3)
    return a2m, statistics.median(ts)

print()
print(f"  {'config':<28}{'pass ms':>9}{'budget':>9}{'frames':>8}{'passes/s':>10}  gate")
for label, ov, steps, online in (
        ("shipped (ov=%d, %d steps)" % (dkw['overlap_v2'], dkw['sampling_timesteps']),
         dkw['overlap_v2'], dkw['sampling_timesteps'], dkw['online_mode']),
        ("online candidate (ov=70,10)", 70, 10, True),
        ("online candidate (ov=40,10)", 40, 10, True),
):
    a2m, ms = measure_pass(ov, steps, online)
    budget = 1000.0 * a2m.valid_clip_len / 25.0
    pps = 25.0 / a2m.valid_clip_len
    ok = "PASS" if ms <= budget else "FAIL"
    print(f"  {label:<28}{ms:9.1f}{budget:9.0f}{a2m.valid_clip_len:8d}{pps:10.2f}  {ok}")
    print(f"  {'':28}lookahead {a2m.valid_clip_len/25.0:.2f}s")

# --- warp + decode per frame ----------------------------------------------
print()
try:
    from core.atomic_components.warp_f3d import WarpF3D
    from core.atomic_components.decode_f3d import DecodeF3D
    warp, dec = WarpF3D(warp_cfg), DecodeF3D(dec_cfg)
    # f_s shape comes from the appearance extractor's real output, not a guess:
    # core/models/modules/appearance_feature_extractor.py reshapes to
    # Bx32x16x64x64 (2.1 M floats = 8.4 MB/frame fp32). WarpingNetwork asserts
    # a 5-D volume, so a [1,256,64,64] guess raises inside deform_input.
    f_s = np.random.randn(1, 32, 16, 64, 64).astype(np.float32)
    kps = np.random.randn(1, 21, 3).astype(np.float32)
    dec(warp(f_s, kps, kps))                     # warm up
    n = 50; t0 = time.perf_counter()
    for _ in range(n):
        out = dec(warp(f_s, kps, kps))
    per = (time.perf_counter()-t0)*1e3/n
    print(f"  warp+decode    : {per:7.2f} ms/frame   budget 40 ms   "
          f"render {out.shape[1]}x{out.shape[0]}   "
          f"{'PASS' if per <= 40 else 'FAIL -> render_size=256'}")
except Exception as e:
    print(f"  warp+decode    : could not measure: {type(e).__name__}: {e}")

if torch.cuda.is_available():
    print(f"  peak VRAM      : {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
          f"of {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
PY

say "Done"
note "Report the four numbers above: RTF (wall/video), LMDM pass ms,"
note "warp+decode ms, peak VRAM. They decide overlap_v2 and render_size."
