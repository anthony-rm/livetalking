# Implementation Plan: Ditto real-time speech at 512, then 14 fps toward the ~17 fps ceiling

## Overview

User decision (2026-11-18): **speech must play real-time regardless of video
fps.** Today, at 14 fps the pipeline emits only 28 audio chunks/s against the
track's 50 Hz consumption, so the pacer splices silence into words and every
utterance stretches ~1.8×. Phase 1 decouples audio from the render rate
(real-time speech by construction) and adds content-locked video frame
dropping — the video-call look: perfect audio, mouth shows the latest rendered
state, mild judder at 14 fps. Phases 2-5 then attack smoothness: probe the
~125 ms/batch host overhead hypothesis, make the LMDM and warp→decode chain
device-resident (14 → ~16-17 fps at 512), and offer cfg/resolution levers on
top.

Targets: sustained speech advances at 100% wall rate (measured); 512 lane
≥16 fps steady at 960p via the device-resident work; identical numerics gates
per change; no regression in drops, A/V offset, or the 32/32 emotion/pcm
tests.

## Non-goals / dead ends (recorded so they are not re-litigated)

| Idea | Verdict |
|---|---|
| INT8 warp/decode | REJECTED at the gate (19.2/20.8 dB vs 40/38, 2026-11-18). Only a structurally different method (per-channel weights, mixed precision, bigger calib set) reopens; not in this plan. |
| Decoder batch sweep (B=1/2/5/20) | Dead: window fixed at 10 by valid_clip_len; B=10 already amortized to 25.9 ms/frame; padding adds latency. |
| Batched/dynamic-batch warp | Dead: B=10 warp measured SLOWER than B=1 (32.3 vs 24.5 ms) — compute-saturated. |
| Custom CUDA GridSample kernels | Deferred: the two torch fp32 grid samples cost ~3-6 ms/frame of a ~50 ms chain; <1 fps upside, numerics risk. Revisit only if B2 shows them costlier. |
| MuseTalk 1.5 swap | Dead on this box: musetalk ~15 fps (unet+vae 55.9 ms/frame), mouth-only motion; user wants Ditto expressiveness. |
| NVENC / 1080p / adaptive resolution / distillation | Out of scope or deferred; see skill references. |

## Architecture decisions

- **Real-time speech is a decoupling change, not a pacing change.** The 50 Hz
  pacing thread, its lead delay line, and the track clock stay as-is. What
  changes: real audio reaches the pacing queue at ARRIVAL rate rather than
  gated at 2-chunks-per-rendered-frame. The tap must sit at the ASR boundary
  — upstream of every drop/throttle — so no word can be lost, and it must
  forward eventpoints (sentence start/end ride userdata and reach the UI via
  notify). `flush_talk` purges the new queue too (barge-in pattern).
- **Video drops content-locked, never audio.** A rendered frame whose paired
  audio window has already been paced out is stale: skip it BEFORE
  paste_back (saves the ~11 ms paste too) instead of queueing it. A bounded
  progress pointer (chunks emitted by the pacer) is the single comparison.
  Stale-frame drop happens in `process_frames`; the track-level
  newest-frame-wins in `recv()` stays as the last line of defense.
- **Env-gated, default = the user's choice.** `LT_AVSYNC=drop` default,
  `=stretch` reproduces today's behavior exactly (fallback + A/B arm).
- **Measure before the GPU rewrite.** The ~125 ms/batch host-overhead number
  is a code-reading hypothesis (per-frame `.cpu()` syncs, ~84 MB f_s H2D,
  ~14+42 MB warp D2H/decode H2D per batch, potential per-step syncs in DDIM).
  B1/B2 probes convert it to tables; Phases 3-5 are sized from those.
- **Wrap, don't fork the vendored tree.** The DDIM port lands repo-side;
  upstream's pinned clone on the box is not edited.
- **Numerics gates, not vibes** — same regime that just caught the INT8 lane.

## Feasibility evidence

| Claim | Source | Status |
|---|---|---|
| Speech stretches today: ~28 real chunks/s vs 50 Hz consumption at 14 fps | `base_avatar.py:528-540` (audio forwarded per rendered frame), `_pace_audio` silence-on-dry; session-4 LT_AUDIO_TRACE measured 23.8 real/s | Measured + verified in code |
| Track tolerates fewer video frames (audio is the pacing clock; video PTS from monotonic elapsed time; newest-frame-wins) | `server/webrtc.py:72-96, 135-158` | Verified in code |
| Engine floor ~59 ms/frame → ~16.9 fps ceiling | LMDM 87.5 ms/pass ÷ 10 (session 7), warp 24.4 (s3), decode 25.9 (s5); both lanes on `torch.cuda.current_stream()` (`trt_warp_hybrid.py:104`, `trt_decode_batch.py:61`) | Components measured; end-to-end ceiling needs B2 |
| Per-frame `.cpu()` sync in warp chain; f_s fp32 H2D; warp D2H + decode H2D per batch | `trt_warp_hybrid.py:189-190`, `_gpu_half` astype(fp32), `TRTDecodeBatch.__call__` from_numpy→copy_ | Verified in code |
| DDIM sampler is numpy, one engine call per step | recorded session 7; per-step sync structure UNVERIFIED (upstream `audio2motion.py` on the box only) — B1 reads + probes it | Partial |
| 313 ms/pass (isolated) vs 172 ms (live trace) vs 87.5 ms (engine sum) | three separate measurements | Incompatible contexts; B1 reconciles |
| lead=10 delay line; lip lag linear in lead (-0.19 s/10 chunks); residual ≈ +0.2 s | session-8 xcorr A/B | Measured; A/V offset must be re-certified after Phase 1 (audio now leads differently) |

## Task list

Recorded as checkboxes in `tasks/ditto-fps/todo.md`. The pre-existing
`tasks/plan.md` is a COMPLETED different plan — do not touch it. Order below
reflects the 2026-11-18 user decision; original task IDs kept to match chat.

### Phase 0 — remaining decisions (T0)

- [ ] **T0 — Answer the remaining open questions** (box access mode,
      deadline, 25fps smoothness target, A/B appetite, lead tuning
      permission). The A/V policy question is ANSWERED: drop-mode default.

### Phase 1 — real-time speech (user priority, first deliverable)

- [ ] **T11a — Audio tap at the ASR boundary.** Real chunks (type=0, with
      userdata) enqueued for the pacer at arrival, parallel to the existing
      pairing path; `process_frames` stops forwarding audio; `flush_talk`
      purges the new queue; synthetic silence stays out (pacer synthesizes on
      dry). Audio timeline now advances at TTS arrival rate (bursts smoothed
      by the pacer's own 50 Hz clock + bounded queue).
      Files: `avatars/base_avatar.py`, `avatars/audio_features/pcm.py` (tap),
      tests for tap ordering + flush purge.
      Gate: sustained speech ≥49 real chunks/s measured (LT_AUDIO_TRACE),
      no word loss vs input WAV (count real chunks == input chunks).
- [ ] **T11b — Content-locked stale-frame drop.** Pacer maintains a monotonic
      emitted-chunk counter; each res_frame_queue item's audio window is
      checked against it; stale → skip paste_back + push. Env `LT_AVSYNC`
      (default `drop`); `stretch` restores the current behavior bit-for-bit.
      Gate at lead as measured after T11a (the delay line semantics change
      when audio leads frames): xcorr offset bounded, non-growing over 60 s,
      |mean| as low as lead tuning allows; suite green (incl. new unit tests:
      drop vs stretch equivalence under a fast fake renderer).
- [ ] **T11c — Certification + fallback check.** Run the box gate (45 s,
      standard protocol) in BOTH modes; record delivered fps, drop counts,
      real-chunk rate, xcorr; confirm `LT_AVSYNC=stretch` reproduces the old
      behavior as the demo fallback.

Checkpoint A: real-time speech shipped and certified at 512/960p. From here
everything is smoothness.

### Phase 2 — decomposition probes (no installs; user runs or live grant per T0)

- [ ] **T1 — LMDM pass decomposition.** CUDA-event time per DDIM step, count
      implicit D2H/H2D per step (read upstream audio2motion.py on the box
      first), cond-window build, a2m() wall vs engine sum; isolated and live.
      Reconciles 313/172/87.5. Sizes Phase 3.
- [ ] **T2 — Render-chain event profile.** Events around A/gs1/B/gs2/C + sync
      + decode + staging, isolated and concurrent with LMDM; quantifies host
      overhead + shared-stream interference. Sizes Phase 4.

Checkpoint B: tables in evidence.md. If host overhead < 40 ms/batch, Phases
3-4 shrink to tuning and effort pivots to Phase 5; if > 100 ms/batch, proceed
as ordered.

### Phase 3 — device-resident LMDM (RESIZED BY T1: value collapsed, priority drops)

- [ ] **T3 — GPU-native DDIM sampler.** Repo-side substitution: resident
      latent/schedule/coefficient buffers, torch elementwise updates, zero
      host round-trips in the step loop. Gate: ≤1e-5 rel diff vs numpy sampler
      on one offline pass (fp16 noise floor) + same-window motion equality;
      update math unit-tested on Mac with a stub engine; env fallback to numpy
      sampler retained.
      **RESIZED BY T1 (measured, 2026-11-18): the numpy DDIM update is 1.1 ms
      of an 87.5 ms pass (1.3%).** Porting the arithmetic to GPU is worth
      ~0.1 fps and is NOT worth doing as an arithmetic port.
      **What IS worth doing, and what T1 actually found:** the per-step
      `TRTWrapper.infer()` does a BLOCKING `cudaMemcpy` D2H of pred_noise and
      x_start, and both threads submit to the SAME default stream — so each
      LMDM step drains the stream and waits behind the renderer's queued work.
      That is why the live lane shows lmdm=172 ms while the isolated pass is
      87.5 ms (2x inflation) and why the live batch (695 ms) exceeds the sum
      of the isolated parts (638 ms) by ~57 ms.
      ⇒ **T3 becomes: give the LMDM its own CUDA stream and make its copies
      async** (cudaMemcpyAsync + events, no blocking D2H), so the 87.5 ms of
      LMDM work hides under the renderer's 550 ms. Target: batch 695 → 570-600
      ms ⇒ 16.5-17.5 fps. Keep the torch-DDIM port as the follow-on (it also
      removes the host round-trip), but only if the stream split lands first.
- [ ] **T4 — CUDA-graph the step loop (conditional).** Only if T1 shows
      residual launch/sync overhead after T3; static buffers are the
      precondition; capture on the box, fall back to T3-only if WSL2/TRT-10
      capture misbehaves. Gate: step-output equality, no regression.
      **Resized by T1: launch overhead is 0.36 ms/step (setup) — graphs are a
      minor lever here; the stream split is the big one.**

### Phase 4 — device-resident warp→decode (RESIZED BY T2: premise mostly wrong)

- [ ] **T5 — Warp chain torch-in/torch-out.** `TRTWarpHybrid` device-resident
      path used by `_gpu_half`; f_s staged to device fp16 once per avatar
      load; numpy contract kept for tests/registration. Gate: mean|diff| vs
      old path at the hybrid's own acceptance scale (0.0002); per-frame `.cpu()`
      syncs gone (T2 re-run).
      **RESIZED BY T2 (measured, 2026-11-18): the render chain ISOLATED
      measures 550.7 ms/batch = 55.1 ms/frame — and the live trace reports
      warp=253 + decode=270 = 523 ms/batch. So the render chain is NOT
      inflated by contention; it is already at its isolated floor. The
      per-frame `.cpu()` syncs in the warp chain cost ~2 ms/frame of a 55 ms
      frame — removing them is worth ~0.4 fps, not the ~2 fps the original
      sizing assumed.**
      ⇒ **T5 becomes: keep the chain as-is, but make its one sync point
      (the warp chain's final `.cpu().numpy()`) async on the renderer's own
      stream** so it doesn't block the inference thread; the real win is
      Phase 3's stream split hiding LMDM under this work.
- [ ] **T6 — Decode consumes warp output on device.** No from_numpy on the
      live path; PNG staging stays async-pinned as-is. Gate: decode output
      bit-identical; batch sync count 12 → 1.
      **Resized by T2: the decode is already a single `execute_async_v3` per
      batch (25.46 ms/frame, one sync) — there is no from_numpy on the live
      path (`_gpu_half` keeps its `f_3d` stack on device). T6 is a no-op;**
      **drop it.**
- [ ] **T7 — Full-lane re-gate.** Standard 45 s protocol, both LT_AVSYNC
      modes; expected ≥16 fps steady at 512/960p; update the serving table +
      skill reference with measured outcomes; browser experience check.
      **Resized: the target is now 16.5-17.5 fps (Phase 3 stream split), not
      the earlier ~18. Re-gate after T3.**

### Phase 5 — cfg levers (quality-gated, each reversible, parallel-able)

- [ ] **T8 — LMDM steps 10→8→6 A/B.** Offline renders of the reference clip;
      user review; env `DITTO_LMDM_STEPS` (default 10).
- [ ] **T9 — overlap_v2 70→65/60 A/B.** Wider valid clip (15/20 frames/pass);
      check fuse quality + TTFF (seed grows with overlap_v2); env
      `DITTO_OVERLAP_V2` (default 70).
- [ ] **T10 — 384 lane (E=48) — CONDITIONAL on T0's smoothness answer.**
      Export r256 pattern at edge 48, decoder_b10 at 384, lane tuples in
      load_model; paste geometry already resolution-generic. Expected 20-24
      fps post-Phase-4; A/B vs 512-downscaled on mouth/eye detail, fps per
      standard protocol. In the drop world this is the smoothness-per-pixel
      lever, increasingly attractive if 512 stays stubborn.

### Phase 6 — measurement hygiene

- [ ] **T12 — Gate-script fps taxonomy.** All four rates (LMDM-emitted,
      rendered-unique, pushed-to-track, client-presented) + p5/stall/max-gap/
      queue-depth per window, so future claims are protocol-locked.

### Final checkpoint

- [ ] Numerics gates green; suite green on pinned interpreter
      (live-server/flaky excluded); both LT_AVSYNC modes certified; decision
      table + skill reference updated; commits staged per-path (siblings
      active).

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Tap changes barge-in/flush semantics (zombie speech in the new queue) | High | flush_talk purges the tap queue + delay line; unit test drives a barge-in and asserts silence within one batch (pattern already exists for feat_queue) |
| Audio leads video differently → lips visibly behind after T11 | Medium | lead is env-tunable; T11b's xcorr arm decides; worst case reduce LT_AUDIO_LEAD toward 0 and re-certify |
| Dropped frames make motion look choppy at 14 fps | Expected (the explicit trade) | Phase 3-5 raise fps; 384 lane is the smoothness escape hatch; stretch mode stays one env away |
| DDIM port subtly changes motion | High | ≤1e-5 gate + user A/B before shipping; env fallback to numpy sampler |
| Probes show host overhead small (plan shrinks) | Medium | That is Phase 2's purpose; pivot with the tables in hand |
| Sibling agents editing ditto_avatar.py concurrently | Medium | git status + re-read before edits; stage only owned hunks |
| Box DNS dead (no installs) | Low | Phases 1-2 need zero installs; engine exports use existing on-box exporters |

## Open questions (remaining, after the A/V answer)

1. Box access mode this round: scripts + you paste outputs, or a live grant
   for the probe sessions (Phases 2-3 need several short box runs)?
2. Deadline for the seminar?
3. Smoothness target now that pacing is solved: is ~16-17 fps at 512
   acceptable judder, or do you want the 384 lane built regardless (T10)?
4. A/B review appetite: how many short clip pairs will you actually eyeball
   (T8/T9/T10 each want one)?
5. Lips lag check at lead=10 in a real browser — still behind? (Tunes with
   T11b, needs your eyes.)
