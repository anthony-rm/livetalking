# Intent: Avatar Video Preprocessing Tool

Status: Confirmed 2026-11-07 (interview-me). Replace this file if the intent changes.

## Outcome
A preprocessing tool that turns a messy talking-head upload into a clean,
standardized vertical (9:16) video of the **same person** — enhanced quality,
corrected lighting, clean flat-color background — which LiveTalking then
consumes as a drop-in base video for any model (wav2lip / musetalk / future).

## User
Operators adding a base avatar video to LiveTalking. Run the tool once per
upload, fully automatic — no knobs, no manual editing per video.

## Why now
Raw uploads vary wildly in quality, background, and lighting, so avatars come
out inconsistent. Standardizing the input before avatar extraction makes every
avatar look good regardless of source clip.

## Success
Feed any low-quality / noisy clip in; get a vertical video with a clean flat
background and visibly better face quality out. It works unchanged as a base
video for wav2lip and musetalk (no changes to their genavatar.py required).

## Constraint
- Fully automatic; no per-video manual tuning.
- Person and background kept **separable internally** (person-matte preserved
  as a sidecar artifact) so backgrounds can be swapped later without re-running
  matting.
- Opaque vertical video is the LiveTalking contract (standard codecs can't carry
  alpha; LiveTalking's genavatar.py already consumes full PNG frame sequences).
- Out of scope: identity / appearance transformation; per-video manual editing;
  true portrait relighting (single-image relight); transparent-output video
  (future); any change to LiveTalking's runtime backends.

## Background strategy
- Default: flat / solid color.
- User can change the background in the app later. Since the person-matte is
  preserved, that is a later feature, not a rewrite.