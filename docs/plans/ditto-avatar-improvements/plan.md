# Implementation Plan: Ditto avatar improvements (2nd round)

Sibling to `tasks/ditto-fps/todo.md`, which stays active for its own open items
(8 unchecked, fps levers). Nothing here touches that plan's scope.

## Overview

Six user asks, cut into five shippable slices. Ordered so the two pure-UI ones
land first (minutes, zero GPU risk), then image-only registration, then idle
motion, then the box-side emotion lab. All local work is verifiable on this Mac;
anything needing Ditto weights or a GPU is a handed-over box gate.

## Architecture Decisions

- **The emotion vocabulary is already complete in the backend.** `avatars/ditto/emotion.py`
  holds all 8 upstream labels and every route parses name / index / list. The gap
  is only `web/index.html`, which hardcodes 3 options. So S1/S2 are UI + one new
  UI-to-route call, not model work.
- **Sticky emotion means MOOD** (decision taken): it colours untagged speech
  windows as well as driving the idle expression. Per-utterance emotion stays as
  the sentence-level tag. Both already exist; the buttons expose the sticky one.
- **Idle motion is a `BaseAvatar` seam, not a Ditto hack.** Add
  `idle_frame(idx) -> frame | None`; the default returns the current behaviour
  (`frame_list_cycle[idx]`), so every other backend is untouched. Ditto overrides it.
- **Idle motion renders through the existing GPU worker.** `_LocalGpuPipeline`
  already owns the single GPU chain; idle batches are submitted like speech
  batches, so there is no second CUDA context and no contention semantics to
  invent. The LMDM is NOT driven for idle (its state machine belongs to the speech
  lane; a concurrent pass would corrupt `kp_cond` / `res_kp_seq`). Idle motion is
  synthesized directly in `x_d` space — breathing sway on pitch/yaw/roll/t/scale
  plus a blink cycle — then stitch → warp → decode.
- **Idle rendering is throttled.** A rendered idle batch is cached and reused for
  `LT_IDLE_REUSE` subsequent silent batches (default 3), so average idle GPU load
  is ~25% instead of pinned at the speech-rate ceiling.
- **Single-image registration needs no artifact-format change.** A 1-frame
  artifact already validates (`frame_count=1`) and the backend already passes
  `is_image_flag=(n_frames == 1)`. What is missing is the upload/path surface
  (route + studio UI) and a test that a 1-frame artifact round-trips.

## Feasibility Evidence

| Claim | Probe | Status |
|---|---|---|
| 8-label vocabulary + parser exist | `avatars/ditto/emotion.py:21-24`, `tests/test_ditto_emotion.py` (green on HEAD) | Verified (source + test) |
| `/set_emotion` route exists and reports the name | `server/routes.py:100-120`, registered at `:410` | Verified (source) |
| UI select offers only 3 of 8 | `web/index.html:110-114` | Verified (source) |
| Silent frames bypass render entirely | `avatars/base_avatar.py:660-662` (`target_frame = self.frame_list_cycle[idx]`) | Verified (source) |
| 1-frame artifacts validate | `avatars/ditto/artifact.py:237-257` (loop over `range(n)`, n=1 legal) | Verified (source) |
| `is_image_flag` wired for `n_frames == 1` | `avatars/ditto_avatar.py:595` | Verified (source) |
| Registration accepts an image path | `references/ditto.md` §6 (`load_source_frames`; upstream smoke test uses `example/image.png`) | Verified (docs, read from upstream source) |
| Idle render quality / GPU cost at 512² | NOT measured — needs the box | Unverified → Phase gate |
| Whether upstream's `_get_emo_avg` normalizes a repeated label into intensity | NOT read yet (lives in `~/ditto-probe`) | Unverified → S5 box read |

## Task List

### Phase 1 — emotion surface (local, no GPU)
- [ ] S1: all 8 emotions in the `/human` select
- [ ] S2: sticky-emotion buttons under the video, wired to `/set_emotion`,
      disabled when the session's backend does not support it

### Checkpoint: Phase 1
- [ ] `pytest tests/` green (live-server suites excluded per AGENTS.md)
- [ ] Local server restarted, page renders, 8 options present, buttons call the route

### Phase 2 — single-image avatars (local + box gate)
- [ ] S3a: route accepts image uploads/paths (`image/*`), docs say so
- [ ] S3b: studio UI accepts an image, label no longer claims "video"
- [ ] S3c: unit test — 1-frame artifact validates and `load_avatar` handles it
- [ ] S3d (USER, box): register one image, connect, confirm motion

### Phase 3 — idle motion (local + box gate)
- [x] S4a: `BaseAvatar.idle_frame()` hook + silent-branch call (default unchanged)
- [ ] S4b: `DittoReal._idle_motion()` — synthetic sway + blink in `x_d` space
- [ ] S4c: idle batches through `_LocalGpuPipeline` with `LT_IDLE_REUSE` throttling
- [ ] S4d: unit tests with a fake warp/dec (no weights)
- [ ] S4e (USER, box): eyeball a single-image avatar idle → speech → idle

### Phase 4 — emotion lab (box script)
- [ ] S5a: script renders a mixture grid to an mp4 gallery
- [ ] S5b: read upstream `_get_emo_avg`; record whether repetition = intensity

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Idle rendering starves the speech lane at speech onset | High | Idle batches are only submitted on `is_all_silence`; the first speech batch stops them immediately. Throttle with `LT_IDLE_REUSE` |
| Synthetic `x_d` motion looks wrong (exp/kp are 63-dim) | Med | Animate only pitch/yaw/roll/t/scale + eye blink; leave exp/kp at the source value. Ship behind `LT_IDLE_MOTION` so it can be switched off without a revert |
| 1-frame registration fails on the box (landmark/eye detection) | Med | S3c tests the artifact path locally; the box gate reports the real failure. Best-frame retry is deliberately out of scope (see Not Doing) |
| Another agent is editing `avatars/base_avatar.py` | Med | Re-read before every edit there; stage only this plan's hunks |
| GPU cost of idle on a demo box | Med | ~25% of the speech-rate cost, and only while silent |

## Not Doing (and why)

- **Stitching several sources into one artifact** — `M_c2o`/`f_s`/identity are per-frame
  and the cycle would jump crop geometry and identity at every seam; upstream never does it.
- **A 9th emotion** — the conditioning head is baked into the LMDM weights and the TRT
  engine's `(1,80,1103)` input; it needs retraining, not wiring.
- **Best-frame retry over several candidate images** — worth it, but it is a
  registration-side feature with its own box gate; keep S3 minimal.
- **LMDM-driven idle** — its state machine is the speech lane's; a concurrent pass
  corrupts `kp_cond`/`res_kp_seq`. Revisit only with a second model instance.
- **Any further fps work** — `tasks/ditto-fps/todo.md` owns that.

## Open Questions

- Idle tempo: is ~8-10 fps enough to read as alive? (box eyeball)
- Should the idle expression follow the sticky emotion immediately, or only from
  the next speech window? (Current S4 design: immediately.)
