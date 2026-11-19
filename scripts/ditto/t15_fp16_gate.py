"""T15 gate: fp16 vs fp32 5-D grid samples in the TRT warp hybrid.

REAL-TENSOR gate (the INT8 lesson: a synthetic randn gate lies in both
directions). Loads the actual avatar artifact + the real hybrid engines and
compares the full warp output per chain, because what matters is whether the
fp16 sampler's rounding survives the segB/segC engines - not the pixel-level
diff of the grid sample in isolation.

Run on the box, server DOWN (one GPU, one engine set):

    DITTO_GRID_FP16 is irrelevant here - the probe drives both lanes
    explicitly through instances. Real tensors come from the avatar
    artifact (f_s volumes, x_s) plus the LMDM-ish x_d spread produced by
    jittering x_s with the upstream motion-space amplitude.

    $HOME/miniconda3/envs/livetalking/bin/python t15_fp16_gate.py \
        [--avatar ditto_real_v2] [--frames 50] [--edge 64]

Verdict printed as PASS/FAIL against the hybrid's own acceptance scale:
mean|diff| <= 2e-4 (0.0002), the scale T5/T13 used. Timing numbers are
reported but NOT part of the gate - contention on a co-tenant box makes
them advisory (re-run on a quiet box before quoting).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

ENGINES_512 = os.path.expanduser("~/ditto-weights/warp_hybrid_engines")
ENGINES_256 = os.path.expanduser("~/ditto-weights/warp_r256_engines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", default="ditto_real_v2")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--edge", type=int, default=64, help="engine feature edge")
    ap.add_argument("--engine-dir", default=None)
    args = ap.parse_args()

    from avatars.ditto import artifact as art
    from avatars.ditto.trt_warp_hybrid import TRTWarpHybrid

    if not torch.cuda.is_available():
        print("FAIL: needs CUDA (this gate runs on the box)")
        return 1

    engine_dir = args.engine_dir or (ENGINES_256 if args.edge == 32 else ENGINES_512)
    avatar_dir = os.path.join("data", "avatars", args.avatar)
    marker = art.validate(avatar_dir)
    n = min(int(marker["frame_count"]), args.frames)
    print(f"avatar {args.avatar}: {n} chains, engine edge {args.edge}")

    # Two instances, one per lane: prime_f_s is a one-shot per instance
    # (second call is a documented no-op), so the fp16 arm needs its own
    # object. Engine weights double in VRAM (~2x a few hundred MB) - fine
    # for a probe run with the server down.
    warp32 = TRTWarpHybrid(engine_dir, device="cuda")
    warp32.grid_fp16 = False
    assert warp32.feature_edge == args.edge, \
        f"engine edge {warp32.feature_edge} != requested {args.edge}"
    warp16 = TRTWarpHybrid(engine_dir, device="cuda")
    warp16.grid_fp16 = True

    def x_d_from(x_s, i):
        """Synthesize a plausible driving pose: source pose plus small
        per-chain jitter in the motion-space amplitude upstream operates
        with (yaw/pitch/roll ~0.01, t ~0.004 - the idle-lane amplitudes,
        which are deliberately small vs speech; scale up for stress)."""
        x_d = {k: np.asarray(v, np.float32).copy() for k, v in x_s.items()}
        rng = np.random.default_rng(1000 + i)
        for key in ("yaw", "pitch", "roll"):
            v = x_d.get(key)
            if v is not None and v.size:
                v.reshape(-1)[0] += 0.03 * float(rng.standard_normal())
        t = x_d.get("t")
        if t is not None and t.size:
            t.reshape(-1)[0] += 0.012 * float(rng.standard_normal())
            t.reshape(-1)[1] += 0.006 * float(rng.standard_normal())
        return x_d

    def run_lane(warp, chains):
        warp.prime_f_s([f.astype(np.float32) for f in f_s_list])
        outs = []
        for idx, x_s, x_d in chains:
            f_3d = warp.run_batch_dev([(idx, x_s, x_d)], device_out=False)
            outs.append(f_3d[0])
        return outs

    # Load the real f_s pool, shrunk to the engine edge like DittoReal does
    from avatars.ditto_avatar import _shrink_f_s_list, _reshape_x_s
    raw = [art.read_f_s(avatar_dir, i) for i in range(n)]
    f_s_list = _shrink_f_s_list(raw, args.edge) if args.edge < 64 else raw
    x_s_list = [_reshape_x_s(art.read_x_s_info(avatar_dir, i)) for i in range(n)]
    chains = [(i, x_s_list[i], x_d_from(x_s_list[i], i)) for i in range(n)]

    # Timing arms (isolated, warmed)
    run_lane(warp32, chains[:10])
    run_lane(warp16, chains[:10])   # warm both lanes
    t0 = time.perf_counter(); run_lane(warp32, chains); t_fp32 = time.perf_counter() - t0
    t0 = time.perf_counter(); run_lane(warp16, chains); t_fp16 = time.perf_counter() - t0

    # Numerics gate: fp16 lane vs fp32 lane, chain by chain
    ref = run_lane(warp32, chains)
    got = run_lane(warp16, chains)
    diffs = np.asarray([
        np.abs(r.astype(np.float32) - g.astype(np.float32)).mean()
        for r, g in zip(ref, got)])
    maxd = np.asarray([
        np.abs(r.astype(np.float32) - g.astype(np.float32)).max()
        for r, g in zip(ref, got)])
    mean_diff = float(diffs.mean())
    worst = float(maxd.max())
    scale = 0.0002
    print(f"mean|diff| = {mean_diff:.3e}  (gate <= {scale})")
    print(f"max|diff|  = {worst:.3e}")
    print(f"per-chain mean|diff| p50={np.percentile(diffs,50):.3e} "
          f"p95={np.percentile(diffs,95):.3e} max={diffs.max():.3e}")
    print(f"timing (advisory, isolated): fp32 {t_fp32*1000:.0f} ms / {n} chains, "
          f"fp16 {t_fp16*1000:.0f} ms -> {(t_fp32/max(t_fp16,1e-9)-1)*100:+.1f}%")
    ok = mean_diff <= scale
    print("VERDICT:", "PASS - set DITTO_GRID_FP16=1 in the box restart env, "
          "then run the 45 s fps gate before/after" if ok else
          "FAIL - keep DITTO_GRID_FP16=0; fp16 sampler rejected on numerics")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
