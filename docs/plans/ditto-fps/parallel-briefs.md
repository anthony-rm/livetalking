# Parallel session briefs — ditto-fps plan, 2026-11-18

How to run the plan with multiple sessions/agents. The GPU box serializes all
box runs (one model instance, `max_session: 1`); local code and Mac-side work
parallelize freely. Each pack below is a self-contained brief: launch a fresh
Hermes session, paste the pack header + its brief, done.

Constants for every pack:
- Repo: /Users/anthonyrm/Projets/livetalking (shared with sibling agents —
  `git status` before ANY edit; stage only your paths; NEVER `git add -A`).
- Plan: tasks/ditto-fps/plan.md, checklist tasks/ditto-fps/todo.md — read both
  before starting. Evidence (tables, gate output) → tasks/ditto-fps/evidence.md
  (append your section; do not rewrite others).
- Skill: `livetalking` — load it. References ditto.md + ditto-runtime-tuning.md
  are the measured-baseline books; do not re-measure what they record.
- Box (access GRANTED 2026-11-18, read + controlled start/stop):
  ssh -p 2222 ecpkn_bac@10.130.104.166 (server on 127.0.0.1:8010, kill-all +
  restart before/during runs as the tapes describe; server currently RUNNING,
  ditto at 512/960p). Never two model instances; check nvidia-smi (at
  /usr/lib/wsl/lib/nvidia-smi, NOT on interactive PATH) before any GPU probe.
- Tunnel for the user browser: ssh -p 2222 -N -L 8010:127.0.0.1:8010 …
  (the plan-holder session runs its own; other sessions do NOT open a
  second tunnel — port conflict).
- Tests: pinned interpreter via ./run.sh resolution; never `python app.py`.
- Measure protocol: phase1_gate pattern (kill-all → fresh restart with
  LT_PYTHON → 45 s WAV → steady windows only; warm-up window excluded).
- Session running the box "owns" the box for its duration — coordinate via
  this file: append "BOX LOCK: <pack> <timestamp>" / "BOX UNLOCK" lines in
  evidence.md under "Box lock log".

## PACK A — audio decoupling (T11a/b/c) — THE critical path

Owner: main session (me). Files: avatars/base_avatar.py,
avatars/audio_features/{base_asr,pcm}.py, server/webrtc.py, tests/.
Goal: real-time speech regardless of video fps (user decision).

Brief for a helper session (see also plan.md Phase 1):
1. Read plan.md §Phase-1, base_avatar.py (`inference`, `_pace_audio`,
   `process_frames`, `render`), base_asr.py, pcm.py, server/webrtc.py in full.
2. T11a audio tap: real audio must reach the pacer at ARRIVAL rate. Design
   (chosen): in LT_AVSYNC=drop mode, `process_frames` stops forwarding audio
   to `audio_out_queue` (that path couples audio to rendered frames — the
   bug); instead `BaseASR.put_audio_frame` (pcm chunks in the ASR queue at
   arrival, type=0, eventpoints in userdata) ALSO enqueues every real chunk
   into a new bounded `asr.audio_tap_queue` (maxsize 400). The pacer drains
   the tap queue. flush_talk purges it AND the pacer's delay line (barge-in
   semantics identical to feat_queue's). LT_AVSYNC=stretch keeps the old
   path bit-for-bit. One subtle point: the thread that fills asr.queue is
   TTS/HTTP-side (put_audio_file / tts.render), and put_audio_frame is
   called there — so the tap adds ~zero work in the ASR worker loop.
3. T11b content-locked drop: pacer maintains monotone emitted-chunk counter
   and stores it on the avatar; each res_frame_queue item (frame, audio_window
   pairs) is checked in process_frames: if the window's chunks have been
   emitted (window end ≤ emitted pointer), skip paste_back + push_video.
   Route selected by env at render() start, not per frame.
4. T12-lite counters: paced_out_chunks, dropped_stale, emitted/stale split,
   log in the existing fps line style.
5. Tests (Mac, pinned interpreter, GPU-free): tap ordering (FIFO), flush
   purge after barge-in, drop decision vs counter, stretch mode unchanged
   behavior, lead semantics in both modes. Follow the existing
   tests/test_pcm_asr.py style; fake renderer classes like those tests use.
6. Gate: LT_AUDIO_TRACE depth trace shows ≥49 real chunks/s during speech;
   render fps unchanged; then box certification run (both modes, standard
   protocol) — see T11c.
Do NOT touch ditto_avatar.py or the TRT modules ( Packs C/D own them).

## PACK B — probes + evidence (T1/T2) — box-locked, read-only on repo

Owner: helper session. Files touched: evidence.md + scratch only.
Runs AFTER Pack A's certification run (needs the server down for isolated
arms and live arms need the shipped A code) — coordinate via box lock.
1. Read plan.md §Phase-2 + ditto.md §8/8b (what to time and what NOT to
   re-measure). Read upstream audio2motion.py ON THE BOX first
   (~/ditto-probe/core/atomic_components/audio2motion.py) — count D2H/H2D
   and sync points per DDIM step from source, then instrument.
2. T1: LMDM pass decomposition — CUDA events around: cond build, engine steps
   (per-step), fuse/cvt, a2m() wall; run isolated (server down) and live
   (LT_STAGE_TRACE run). Reconcile 313/172/87.5 in evidence.md.
3. T2: render-chain profile — events around warp A/gs1/B/gs2/C + sync + .cpu(),
   decode + staging; isolated vs concurrent-with-LMDM arms. Quantify host
   overhead + interference → re-size Phases 3/4 in the plan (EDIT plan.md
   §Phase-3/4 sizing notes, labelled "resized by T1/T2").
4. Scripts live in scratch (pipe-over-ssh pattern: `ssh … 'bash -s' <
   probe.sh > out.txt`); raw output files also to scratch, distilled tables
   to evidence.md.

## PACK C — device-resident LMDM (T3, T4 conditional)

Owner: helper session. Files: avatars/ditto/ (new module, e.g.
avatars/ditto/lmdm_torch.py), wiring in avatars/ditto_avatar.py load_model
(hunks co-owned with Pack D — see collision rule), tests.
START after T1 lands (sizing), or in parallel if you accept the risk that T1
shrinks it. Box needed for the numerics gate (one offline pass).
1. Read plan.md §Phase-3, ditto.md §3/§4 (Audio2Motion state contract),
   upstream audio2motion.py on the box (KP_COND, res_kp_seq, fuse_alpha,
   _smo window — state that must survive the port).
2. Implement the DDIM update as torch ops with resident buffers: latent,
   noise schedule (alphas_cumprod etc. precomputed per cfg), step math
   elementwise; zero .cpu()/.numpy() in the loop; keep a numpy fallback
   path via env DITTO_LMDM_SAMPLER=torch|numpy (default numpy until gated).
3. Numerics gate: one offline pass on the box, rel-diff vs numpy sampler
   output ≤1e-5 on res_kp_seq (fp16 noise floor); unit tests on Mac with a
   stub engine for the update math.
4. T4 CUDA graphs: only if T1 shows residual launch/sync overhead after T3;
   capture with static buffers, verify step-output equality, fall back
   cleanly (env DITTO_LMDM_GRAPH=1) if WSL2/TRT-10 capture misbehaves.

## PACK D — device-resident warp→decode (T5, T6, T7)

Owner: helper session. Files: avatars/ditto/trt_warp_hybrid.py,
avatars/ditto/trt_decode_batch.py, avatars/ditto_avatar.py (_gpu_half +
_stage-related hunks), tests.
START after T2 (sizing). Highest expected win (~2 fps at 512) but most
invasive; runs its box gate via coordination.
1. Read plan.md §Phase-4 + the two TRT modules + ditto_avatar._gpu_half
   (lines ~594-631) first.
2. T5: torch-in/torch-out path on TRTWarpHybrid (mode kept for the
   numpy contract; internal sync structure preserved, final .cpu() removed
   for the live path — one event at chain end; f_s staged to device fp16
   once per avatar load in DittoReal.__init__; numpy contract intact for
   tests + registration flows).
3. T6: TRTDecodeBatch callable path consumes the on-device warp output
   (the np.stack stays but operates on torch stack) — no from_numpy on the
   live path; PNG staging via _stage_batch_gpu unchanged.
4. Gates: mean|diff| vs old path ≤ hybrid's own 0.0002 scale on 50 real
   frames; sync count 12 → 1 (count via event instrumentation or trace);
   full-lane gate after (T7 protocol; both LT_AVSYNC modes; expected
   ≥16 fps at 512/960p) → update the serving table in the SKILL
   reference + evidence.md.

## PACK E — 384 lane (T10) + cfg knobs (T8/T9)

Owner: helper session. Files: scratch (engine exporter runs on the BOX),
avatars/ditto_avatar.py lane tuples (shared with C/D — collision rule below),
avatars/ditto/genavatar.py (no — read-only), env knobs in ditto_avatar.py.
1. Read plan.md §Phase-5 + ditto-runtime-tuning.md §2 (what NOT to re-try).
2. T10 engines: adapt scratch/ditto/export_r256.py pattern to E=48 (warp
   segA/B/C 32→48 edge; decoder_b10 at 384) via a small box script; verify
   numerics vs pytorch at 384 the same way r256 was (mean|diff| + PSNR vs
   512-downscaled on ~50 frames); then lane validation: DITTO_RENDER_SIZE
   accepts 384 and the r384 engine dir auto-detect, shape-guards raise
   loudly on edge mismatch (the 20 fps-of-noise precedent).
3. T8/T9 knobs: DITTO_LMDM_STEPS (default 10) and DITTO_OVERLAP_V2 (default
   70) env passthrough in ditto_avatar ONLINE_KWARGS construction + a
   short note in the skill when a non-default proves out.
4. Box gates for each A/B arm as standard. Quality review by the user on
   rendered clip pairs (they said they can eyeball "a lot").

## Collision rules (all packs)

- ditto_avatar.py is co-owned by C, D and E. Rules: (1) each pack's hunks are
  named in its brief; (2) before each patch, `git diff avatars/ditto_avatar.py`
  — if foreign hunks appeared, rebase your edit onto the fresh read, or
  coordinate in evidence.md's "Coordination log" section; (3) commit each
  slice atomically with a pack-prefixed message (a11: …, c3: …).
- engine dirs on the box: nobody touches ~/ditto-weights/warp_int8_engines
  (rejected lane, kept as evidence) or the fp16 lanes; new lanes get their
  own dirs (warp_r384_engines).
- The plan.md sizing notes for Phases 3/4 are re-sized BY PACK B ONLY (T1/T2
  authors), labelled "resized by T1/T2".
- One box lock at a time; the lock log lives in evidence.md.

## What the user runs (eyeball lane)

- Tunnel + browser: the main session keeps the tunnel up (proc id in
  evidence.md); open http://127.0.0.1:8010/index.html, avatar ditto_real_v2,
  Connect. Send long-ish text; listen for stretching (current) vs real-time
  (after A ships). Admin at /admin.html.
- A/B eyeball reviews: each pack posts clip pairs in evidence.md with exact
  commands to reproduce; user watches and votes in chat.

## Sequencing summary

- Now: A (critical path, local code) ∥ no box contention.
- A ships + certifies → B (probes, needs A's code running) ∥ C can start
  beside B (risk: T1 may resize it) ∥ E can start its export work (box
  locked windows).
- B lands tables → C and D fully resourced (D is the biggest win).
- D lands → re-gate (T7) on the new chain, update the books.
- E lands last (lanes are additive) or after D if the smoothness vote wants
  384 as the default.
- A's own box certification (T11c) is the first box lock after its coding.
