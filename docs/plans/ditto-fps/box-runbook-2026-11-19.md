# T9/T8/T7/T15 box runbook — one page, five commands

Per AGENTS.md the box is USER-run; this page is the exact command list to
paste. Everything else (lever code, gates, protocol) is already shipped.
Two scripts carry the work: `scripts/ditto/t15_fp16_gate.py` (repo) and
`ab_arm.py` (this scratch file, scp'd below).

Arms are ONE 45 s gate each, same synthetic-speech WAV, same avatar
(`ditto_real_v2`), 512/960p/H264 (the serving config, already pinned in the
box's restart.sh). fps numbers cited below are the server's
`actual avg final fps` line — the client counter prints the audio clock.

Pre-checks (once, before any arm):

    ssh -p 2222 ecpkn_bac@10.130.104.166
    systemctl --user is-active qyy-pb06-dialogue qyy-pb06-rtc qyy-pb06-relay || true
    /usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv

If a qyy-pb06 unit is active, note it next to every number (contention) or
stop it for the window (your co-tenant call, not the agent's).

## 1. Copy the two scripts to the box (from the Mac)

    scp -P 2222 scripts/ditto/t15_fp16_gate.py \
        ecpkn_bac@10.130.104.166:/tmp/t15_fp16_gate.py
    scp -P 2222 /Users/anthonyrm/.hermes/profiles/livetalking/cache/scratch/ab_arm.py \
        ecpkn_bac@10.130.104.166:/tmp/ab_arm.py

## 2. T7 re-gate + T9 in one sweep (four arms, ~6 min each)

    for OV in 70 65 60; do
      echo "===== DITTO_OVERLAP_V2=$OV ====="
      ssh -p 2222 ecpkn_bac@10.130.104.166 \
        "cd ~/livetalking_box && env LT_PYTHON=\$HOME/miniconda3/envs/livetalking/bin/python \
         DITTO_OVERLAP_V2=$OV AVATAR=ditto_real_v2 SECONDS=45 LOG=/tmp/ab_ov$OV \
         \$HOME/miniconda3/envs/livetalking/bin/python /tmp/ab_arm.py"
    done

- The first arm (OV=70) IS the T7 re-gate: same binary as the 17.66-17.96
  night gate but with b16 (content-locked pick) and the idle-clear fix in —
  if it lands ≥17, b16 costs nothing at fps level. Arms 65 and 60 answer T9.
- Paste the three `--- server fps windows ---` blocks back to the agent.

## 3. T8 — LMDM steps 10 -> 8 / 6 (pick AFTER the T9 winner is known)

    ssh -p 2222 ecpkn_bac@10.130.104.166 \
      "cd ~/livetalking_box && env LT_PYTHON=\$HOME/miniconda3/envs/livetalking/bin/python \
       DITTO_OVERLAP_V2=<the 65-or-60 winner> DITTO_LMDM_STEPS=8 \
       AVATAR=ditto_real_v2 SECONDS=45 LOG=/tmp/ab_steps8 \
       \$HOME/miniconda3/envs/livetalking/bin/python /tmp/ab_arm.py"

then the same with DITTO_LMDM_STEPS=6 (LOG=/tmp/ab_steps6). Fewer DDIM steps
is a quality lever first (coarser sampling) — eyeball BEFORE trusting the
fps delta, which is expected small (~0.1-0.3) since the LMDM is overlapped.

## 4. T15 — fp16 grid samples: numerics gate THEN fps gate

Numerics (server DOWN — it builds its own engine instances):

    ssh -p 2222 ecpkn_bac@10.130.104.166 \
      "cd ~/livetalking_box && DITTO_WARP_ENGINE_DIR= \
       \$HOME/miniconda3/envs/livetalking/bin/python /tmp/t15_fp16_gate.py \
       --avatar ditto_real_v2 --frames 50"

PASS = mean|diff| <= 2e-4 vs the shipped fp32 sampler on real artifact
tensors. If (and only if) it prints PASS:

    ssh -p 2222 ecpkn_bac@10.130.104.166 \
      "cd ~/livetalking_box && env LT_PYTHON=\$HOME/miniconda3/envs/livetalking/bin/python \
       DITTO_GRID_FP16=1 <T9 winner overlap env if kept> \
       AVATAR=ditto_real_v2 SECONDS=45 LOG=/tmp/ab_grid16 \
       \$HOME/miniconda3/envs/livetalking/bin/python /tmp/ab_arm.py"

FAIL leaves DITTO_GRID_FP16=0 (the default) — the lever is then dead on
this silicon, recorded as such.

## 5. S3d/S4e — the two eyeballs (no fps gate; your eyes only)

S3d (still-image registration): in Avatar Studio register a .jpg with
model=ditto, connect, send one sentence, confirm mouth + audio.

S4e (idle breathing on that same avatar): after the sentence ends watch
idle -> the synthesized breathing loop takes over (first ~50 frames render
on demand, then the loop is cached and idle is GPU-free). Watch for:
breathing too subtle/too large, the first idle frame's latency, whether
LT_IDLE_REUSE=4 (default) stutters. `LT_IDLE_REUSE=2` doubles the idle
frame rate if 4 looks steppy; `LT_IDLE_MOTION=0` restores the still image.

## 6. Settle the winner

Tell the agent the numbers; the box's restart.sh export line gets the
winning env (DITTO_OVERLAP_V2=…, optionally DITTO_LMDM_STEPS=…,
DITTO_GRID_FP16=1), and the docs' decision tables get MEASURED rows
(sourced from pasted gate output, never from memory).

## Fuse-seam eyeball (T9's quality gate) — what to look for

With overlap 65/60 the LMDM emits 15/20 frames per pass instead of 10 and
crossfades a proportionally wider band at each window seam. Seams land
every ~0.6/0.8 s of speech (15/20 frames at 25 rows/s). In the browser
watch one long sentence at each OV value and look for a repeating
micro-pause or motion "reset" at that cadence — a subtle hitch that returns
on rhythm. If either arm shows it, keep 70 even if its fps is lower; the
skill's earlier "65 → 18.4-18.8" rows were fabricated (2026-11-19 audit) —
nothing about 65/60 is known until these arms actually run.
