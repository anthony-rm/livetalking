# Evidence: ditto-fps plan

## Baseline (pre-plan, measured — cites recorded sessions)

- 512 lane steady: 13.8–14.0 fps at 960p/h264, ~2 drops/window (~0.15%),
  GPU 89% util / 61.7 C / 5.2 GiB during speech (session-9 gate, cb46bfe).
- Engine floor: LMDM 87.5 ms/pass ÷ 10 frames; warp 24.4; decode 25.9
  ms/frame → ~59 ms/frame → ~16.9 fps shared-stream ceiling (plan §evidence).
- Speech stretches today: ~28 real chunks/s reach the track vs 50 consumed
  (2 chunks forwarded per rendered frame at 14 fps; session-4 trace measured
  23.8 real/s).
- INT8 lane: REJECTED 19.2/20.8 dB PSNR vs 40/38 gate (2026-11-18).

## Box lock log

(append: "BOX LOCK: <pack> <UTC time> <what>", "BOX UNLOCK: <pack>")

## Coordination log

(append-only; packs record collisions/deferrals here)

- 2026-11-18 main session: tunnel running locally (pid ~89506,
  ssh -p 2222 -N -L 8010:127.0.0.1). Server on box: RUNNING, ready,
  ditto 512/960p. User browser: http://127.0.0.1:8010/index.html
- 2026-11-18 14:4x CST takeover (Hermes session 2, the one the user is
  talking to directly): found base_avatar.py syntactically broken
  (IndentationError at process_frames, line ~537) since 12:17 with no
  coordination entry; base_asr.py tap landed cleanly. Per "no deadline, go
  how far you can", this session resumes Pack A ownership and completes
  T11a/T11b itself. Original tunnel pid gone (replaced: this session's
  tunnel is live, pid 17723). If the earlier session is still running and
  mid-edit on base_avatar.py: STOP — the file is broken; coordinate here
  before any further edit to that file.

## Probe output sections

### T1/T2 — decomposition probes (2026-11-18, box, server DOWN, 512 lane)

Scripts: `scratch/t1_lmdm_probe.py`, `scratch/t2_render_probe.py` (also on the
box at /tmp). Probes: `t1_lmdm_probe.py` (LMDM pass, per-step split),
`t2_render_probe.py` (warp + decode + whole chain).

**T1 — LMDM pass, isolated (12 passes, median):**

| stage | ms/pass | ms/step |
|---|---|---|
| total wall | **87.5** | 8.75 |
| `infer()` (execute_async_v3 + BLOCKING D2H of pred_noise/x_start) | 82.7 | 8.27 |
| `setup()` (host→engine binding, per step) | 3.6 | 0.36 |
| **numpy DDIM update** | **1.1** | 0.11 |

**The numpy sampler the audit wanted ported to GPU is 1.3% of the pass.**
Porting it (+ CUDA graphs) buys ≈5 ms/pass ≈ 0.1 fps — not the 2 fps implied.
The audit's §1.2 diagnosis was directionally right (there IS a host↔device
boundary per denoising step) but the cost is in the blocking D2H that
serialises the stream, not in the numpy arithmetic.

**T2 — render chain, isolated (median):**

| stage | value |
|---|---|
| warp hybrid, per frame | 28.7 ms |
| decode B=10 | 254.6 ms (25.46 ms/frame) |
| **full chain per 10-frame batch** | **550.7 ms = 55.1 ms/frame** |
| chain-only ceiling | **18.2 fps** |

**THE ARITHMETIC — the 14 fps is NOT a compute ceiling.**

| quantity | ms/batch | ms/frame |
|---|---|---|
| isolated LMDM (T1) | 87.5 | 8.75 |
| isolated render chain (T2) | 550.7 | 55.1 |
| **sum of the two** | **638.2** | **63.8** → 15.7 fps if serialised |
| live lane (LT_STAGE_TRACE) | 695 | 69.5 → **14.0 fps** |

So the live lane loses ~57 ms/batch to serialisation, and the LMDM's 87.5 ms
of REAL work is not hidden under the renderer's 550 ms of work even though
the two threads are nominally concurrent: they submit to the SAME default
stream, and `TRTWrapper.infer()` does a BLOCKING `cudaMemcpy` D2H per step,
which drains the stream — so each LMDM step waits behind the renderer's
queued work. That is the mechanism behind the live lmdm=172 vs isolated 87.5
(2x inflation) and it is why the audit's max(31,50)=20 fps never materialised.

**Ceilings, corrected (2026-11-18 late — updated after b13/b14 shipped):**
- chain alone (legacy sync lane): 18.2 fps
- chain alone (run_batch device lane, T13 isolated): 457 ms/window → 21.9 fps
- chain + LMDM overlapped: b13 measured the overlap is NOT free — SM
  contention eats ~2/3 of the win (14.0 → 14.7, not →18)
- b14 (run_batch, shipped): live 15.1-15.4 fps at 512/960p
- 25 fps at 512 is NOT reachable in this renderer family on this silicon.
  Post-b14 the renderer's remaining time is pure kernel compute (warp
  engines 28.7 ms×10 + two fp32 grid samples + decode 25.5 ms/frame);
  scheduling is exhausted.

**b13 — LMDM private stream (commit ce90cf8):** `LT_LMDM_ASYNC=1` (default).
T12 probe (equality + loaded interference): bit-identical; pass 494→186 ms
under render load. Live 14.0→14.7.

**b14 — run_batch device-resident warp (commit b961fac):** T13 probe:
bit-identical (max|diff| 0.0); window 522→457 ms isolated. Live 14.7→15.1-15.4.
Stage trace after b14: warp 333 + dec 250 ms — kernels, not waits.

**Measurement hygiene (important, discovered here):** the box hosts a SECOND
project (`~/qyy/pb06-avatar`, systemd user units `qyy-pb06-{dialogue,rtc,relay}`):
Qwen3-1.7B LLM service (GPU-resident) + a `app.realtime profiles/realtime-serena.yaml`
wav2lip/Qwen3-TTS service that RESPAWNS every ~2-3 min. With OUR server
stopped, the GPU still holds 6.5 GB/12 GB and the box's CPU is busy. All fps
numbers in this program are therefore contended unless those units are
paused — a fact to state in every measurement (and to ask the user about
before running long benchmarks).

### T11a/T11b/T11c — drop-mode certification (2026-11-18, box)

Server: `LT_AVSYNC=drop LT_AUDIO_TRACE=1`, `ditto_real_v2`, 512 lane,
960p/h264. Drive: `phase1_gate.py`, 6 s WAV via `/humanaudio`, 45 s run.

| metric | drop mode | stretch mode (baseline) |
|---|---|---|
| real chunks emitted by the pacer | **2328** | ~84 (14 fps x 6 s) |
| dry ticks (silence spliced into speech) | **0** | 40–70% of ticks |
| delivered video fps | 13.57–13.80 | 13.8–14.0 |
| dropped frames/window | 2 | 2 |
| stale-skip events | 0 (removed) | — |

**Audio is real-time by construction: 50 chunks/s, zero silence spliced.**
Video is unchanged — the render is still the render. The residual cost of
drop mode is a lip-sync LAG that grows during sustained speech and recovers
in pauses (bounded by render speed, not policy); `LT_AUDIO_LEAD` is what
compensates it. `LT_AVSYNC=stretch` reproduces the old behaviour bit-for-bit
(env pin in the pacing/lead tests).

Two policies were tried and REJECTED by measurement, recorded so they are
not re-tried:
1. **Content-locked frame drop** (drop frames whose audio window is behind
   the play pointer): with real-time audio the pointer runs at 50 chunks/s
   while the renderer produces mouth content at only ~28 chunks/s (14 fps x
   2 chunks), so after ~1 s of speech every rendered frame is already
   "behind" and the check freezes the picture.
2. **Queue-depth drop** (drop if a newer frame is queued): the render queue
   holds a whole 10-frame batch, so this discarded 9 of every 10 frames and
   starved the video track entirely (measured: no `actual avg final fps` line
   for the whole arm). Both were replaced by "push every rendered frame,
   let the WebRTC track's newest-frame-wins decide".

Tests: `tests/test_avsync_drop.py` (9/9), plus the pacing/lead tests pin
`LT_AVSYNC=stretch` so the default cannot break them. Full suite: 148
passed; the only 2 failures are the documented load-flaky 50 Hz assertions
in `test_audio_pacing.py` (reproduced on pristine HEAD: 30 chunks/s on
this machine).
