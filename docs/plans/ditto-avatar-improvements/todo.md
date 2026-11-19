# Task List: Ditto avatar improvements (2nd round)

Ordered checklist. One item per task in `plan.md`. Check off only when the
acceptance criteria and verification pass. Sibling plan `tasks/ditto-fps/`
remains active for its own items.

## Phase 1 — emotion surface (local, no GPU)
- [x] S1 — all 8 emotions in the `/human` select (`web/index.html`).
      DONE: the select is now rendered from `GET /api/emotions`, so the label
      order lives only in `avatars/ditto/emotion.py` (upstream's own order).
      A hardcoded list in the page was the actual defect; re-typing 8 options
      would have re-created it.
- [x] S2 — sticky-emotion buttons under the video stream.
      DONE: one button per emotion + Default, POSTing `/set_emotion`; the active
      button reflects the server's answer; disabled when the response says
      `supported: false` (non-ditto backend) or before a session exists.
      Verified live: `/api/emotions` serves 8 labels, `/set_emotion` on a
      missing session answers `session not found`, `index.html` 200.

## Checkpoint: Phase 1
- [x] `pytest tests/` green (183 passed, live-server suites excluded).
- [x] Local server restarted; page renders; routes answer from evidence
      (`/api/model/status` JSON, `/api/emotions` JSON, pidfile present).

## Phase 2 — single-image avatars
- [x] S3a — `server/avatar_routes.py` accepts an image upload/path for ditto.
      DONE: no extension gate added (the generator decides); the requirement
      message now names all three inputs and the docstring records that a still
      image goes through the same registrar as a video.
- [x] S3b — `web/avatar.html` accepts `image/*` and says so.
- [x] S3c — 3 unit tests: a 1-frame artifact validates, is detected as ditto,
      and still fails loudly when frame 0 is missing.
- [ ] S3d (USER, box) — register one image, connect, confirm the avatar speaks.
      Command: `LT_PYTHON=$HOME/miniconda3/envs/livetalking/bin/python
      ./restart.sh <args>` then use Avatar Studio with a .jpg and model=ditto.

## Phase 3 — idle motion
- [x] S4a — `BaseAvatar.idle_frame(idx)` hook; the silent branch calls it and
      falls back to `frame_list_cycle[idx]`. Default behaviour byte-identical.
- [x] S4b — `DittoReal._idle_motion(i)` — breathing sway on pitch/yaw/roll/t
      and scale; `exp`/`kp` deliberately untouched (see plan's Not Doing).
- [x] S4c — idle renders ride the ONE GPU worker via `_LocalGpuPipeline
      .run_on_gpu` (engine static buffers stay single-owner), frames cached per
      loop index, `LT_IDLE_REUSE` sets the displayed rate. After one full loop
      idle playback costs no GPU.
- [x] S4d — 12 unit tests (fake warp/dec, no weights): default hook, opt-out for
      video artifacts, `LT_IDLE_MOTION=0`, pose moves while exp/kp do not, no
      artifact mutation, periodicity, throttle reuse, per-index caching, frames
      differ, render failure degrades to the still image.
- [ ] S4e (USER, box) — eyeball idle → speech → idle on a single-image avatar.
      Watch for: breathing too subtle/large, first idle frame latency (~50
      renders before the loop is warm), and whether a 4-frame reuse stutters.

## Phase 4 — emotion lab
- [x] S5a — `scripts/ditto/emotion_lab.py` renders a mixture grid to one mp4
      (label burned in, per-setting gaps). Parses + `--help` verified locally;
      needs the box to run.
- [x] S5b — ANSWERED (read on the box, 2026-11-19). Repeating a label is a
      NO-OP: `_get_emo_avg` does `emo_avg[i] = 8` (assignment, not
      accumulation) then `softmax`. So `[happy]`, `[happy, happy]` and
      `[happy]*3` all give happy=0.997657. Mixing DOES work, but only as an
      unweighted mean: `[happy, neutral]` and `[happy, happy, neutral]` both
      give 0.499497/0.499497.
      Source: `~/ditto-probe/core/atomic_components/condition_handler.py:6-16`.
      **Consequence for the thesis:** there is no intensity knob in the label
      API, and `softmax(8·one_hot)` is already saturated at 0.9977 — so the
      model sees an effectively hard one-hot for single labels and a 50/50
      blend for pairs, with nothing in between. Task 10's valence-arousal
      mapping must produce its own soft 8-vector rather than approximating
      one from a label list.
