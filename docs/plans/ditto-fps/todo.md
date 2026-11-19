# Task List: Ditto real-time speech at 512, then 14 → ~17 fps

Ordered checklist. Full detail in `plan.md` (same dir). Last updated
2026-11-18 late — reflects shipped work through commit b961fac.

## Phase 0 — decisions
- [x] T0 — ANSWERED by user: no deadline ("go how far you can"), no quality
  gate ("the higher the better"), `LT_AVSYNC=drop` default, **384 lane
  REJECTED** (do not build), box access granted live, A/B = eyeball clips.

## Phase 1 — real-time speech
- [x] T11a — Audio tap at ASR boundary (commit a11). Certified: 2328 real
      chunks over 45 s, 0 dry ticks, 50 chunks/s regardless of render fps.
- [x] T11b — Frame-drop policy: the planned content-locked drop was
      REJECTED by measurement (freezes during sustained speech; queue-depth
      variant starves the track). Shipped policy: video unchanged, WebRTC
      newest-frame-wins; `LT_AVSYNC=stretch` = old behavior (fallback).
- [x] T11c — Box certification (a11): 13.57-13.80 fps delivered initially,
      audio contract intact in both modes.

## Phase 2 — decomposition probes
- [x] T1 — LMDM pass decomposition (tables in evidence.md §T1; run twice,
      quiet-box reproduction within variance). Key result: numpy DDIM update
      is 1.1 ms of an 87.5 ms pass; blocking per-step D2H is the real cost.
- [x] T2 — Render-chain profile (evidence.md §T2): 550.7 ms/batch isolated,
      chain-only ceiling 18.2 fps; live-vs-isolated delta is contention.

## Phase 3 — device-resident LMDM
- [x] T3 — RESIZED and SHIPPED as b13: LMDM private CUDA stream +
      infer_async, per-step sync scoped to it, setup() primed once (was
      free/malloc-ing per step). T12 probe: bit-identical; loaded-interference
      494 → 186 ms/pass. Live: 14.0 → 14.7 fps.
- [ ] T4 — CUDA-graph step loop: REPRICED DOWN (launch overhead 0.36 ms/step)
      — only worth it bundled with a warp-chain graph, see T13b below.

## Phase 4 — device-resident warp→decode
- [x] T5 — RESIZED and SHIPPED as b14: `TRTWarpHybrid.run_batch` — all 10
      chains one stream, ONE sync per window at decode, decode consumes
      device tensors (no D2H/re-upload). T13: bit-identical, window
      522 → 457 ms. Live: 14.7 → 15.1-15.4 fps. REMAINDER: f_s is still
      H2D'd per chain per batch (80 MB/batch pageable) — see T14.
- [x] T6 — dropped (no-op; decode was already one call/batch).
- [x] T7 — re-gate after b14: 15.1-15.4 fps @512/960p, 3 drops/window,
      50 chunks/s, 0 dry. Skill reference updated.

## Phase 5 — cfg levers (quality-gated, reversible; arms pending user box time)
- [ ] T8 — LMDM steps 10→8/6 A/B (`DITTO_LMDM_STEPS`). Value now mostly
      contention relief (LMDM is overlapped): ~0.1-0.3 fps. Needs user eyes.
      Run order + commands: box-runbook-2026-11-19.md §3 (after T9).
- [ ] T9 — overlap_v2 70→65/60 A/B (`DITTO_OVERLAP_V2`). THE biggest
      remaining lever: pass cost (~90 ms) is fixed per window; valid=15/20
      spreads it over 1.5-2x the frames → 9 → 6-4.5 ms/frame of LMDM work,
      less SM contention with the renderer. Quality gate: fuse-seam motion.
      **STATE 2026-11-19 (audit): never gated at 65/60.** The 65/60 arms
      were launched on the box the night of 2026-11-18 but the session died
      before any gate ran; the "18.4-18.8/18.8-19.0" numbers later written
      into the skill were fabricated and purged. Only the 70 arm is
      measured: 17.66-17.96 (b15 night gate, 4 steady windows, pre-b16).
      Commands: box-runbook-2026-11-19.md §2.
- [ ] ~~T10 — 384 lane~~ DEAD (user decision 2026-11-18).

## Phase 6 — measurement hygiene
- [x] T12 — partially: per-second fps-regime probe exists
      (`box_fps_series.py`); the full 4-rate taxonomy in phase1_gate not
      folded in. Low value now that per-second data exists.
- [x] T14 — SHIPPED as b15 (687508e): `TRTWarpHybrid.prime_f_s` stages the
      cyclic f_s pool (75 frames × 8.4 MB fp32) on device once per avatar
      load — 629 MB at edge 64 — instead of H2D-ing ~84 MB/batch pageable and
      re-casting to fp32 on the CPU thread. `run_batch_dev` consumes the pool.
- [ ] T15 (new, optional) — fp16 grid samples in the warp hybrid (currently
      forced fp32): ~0.2-0.5 fps; needs the 0.0002-scale numerics gate.
      **BUILT 2026-11-19:** `DITTO_GRID_FP16=1` opt-in (default fp32) in
      `avatars/ditto/trt_warp_hybrid.py`; real-tensor gate
      `scripts/ditto/t15_fp16_gate.py` (mean|diff| ≤ 2e-4 vs fp32, full
      chain output, real avatar tensors, timing advisory). Box procedure:
      box-runbook-2026-11-19.md §4 — numerics first (server down), fps arm
      only on PASS.

## Corrections to this plan's status (verified 2026-11-19)

The boxes below were written when the plan was authored. These were re-checked
against the working tree:

- **T9's lever is BUILT but never gated.** `DITTO_OVERLAP_V2` exists
  (`avatars/ditto_avatar.py:65`, default 70) and b15 wired it, but no A/B
  result is recorded anywhere. The plan calls it "THE biggest remaining
  lever" — it is still unexercised. **Evening addendum: "65 → 18.4-18.8 /
  60 → 18.8-19.0" rows that briefly appeared in the skill reference were
  fabricated (no gate, no log, no terminal call) — detected and purged the
  same day. The 70-arm only: 17.66-17.96 fps (b15 night gate, 4 steady
  windows, pre-b16).**
- **T7's fps number is stale.** The certification below says 15.4 fps at
  b14, but b15 (f_s staging), b16 (content-locked picker) and the
  idle-clear fix all landed after it. A re-gate is the only way to report a
  current number — the OV=70 arm of the runbook sweep doubles as it.
- **T2's "chain-only ceiling 18.2 fps" is measured, T3/T5 ceilings are not
  re-derived** after b15 — treat 15.4 as a lower bound, not a ceiling.

## Final checkpoint
- [x] Numerics gates green (T12/T13 equality arms); suite green on pinned
      interpreter (148 passed; 2 documented load-flakes excluded from
      blocking); drop mode certified; skill + decision tables updated;
      commits per-path (a11, ce90cf8, b961fac).
- [ ] User eyeball at 15.4 fps (server up, 512/960p).
- [ ] Optional levers T8/T9/T14/T15 if the user wants the last ~1-1.5 fps.
