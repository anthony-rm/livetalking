# Plans — index and current state

All planning documents live here. Every status below was re-verified against the
working tree on 2026-11-19, not copied from an earlier summary.

| Plan | State | Remaining |
|---|---|---|
| [enhance-tool/](enhance-tool/) | **Complete** — 8/8 tasks, 8 tests | nothing |
| [ditto-fps/](ditto-fps/) | Shipped through b14/b15; levers built | T8, T9-A/B, T7 re-gate, T15 |
| [ditto-avatar-improvements/](ditto-avatar-improvements/) | 14/15 shipped | S3d, S4e (box eyeballs) |
| [thesis/](thesis/) | Phase 0a done, 0b partial — **gitignored** (PII) | supervisor asks; 2027 phases |
| [ditto-architecture.md](ditto-architecture.md) | Historical design record | — |
| [ditto-sessions.md](ditto-sessions.md) | Lab notebook, session-9 newest | append per session |

`ditto-architecture.md` and `ditto-sessions.md` were `PLAN_DITTO.md` and
`PLAN_DITTO_NEXT.md` at the repo root; `tasks/` held the rest. All of it is now
here, and `tasks/` is gone. Both are kept because code comments cite their
section numbers (`PLAN_DITTO.md §8.3` → `ditto-architecture.md §8.3`).

`thesis/` is gitignored because it names the student ID, advisor and school
(see `.gitignore`); its plan holds the same details, so it is ignored too.

## What shipped since the plans were last updated

| Commit | Change | Plan item |
|---|---|---|
| `b15` 687508e | device-resident `f_s` pool + overlap lever | T14 **done** |
| `b16` ab90e83 | content-locked motion pick | T11b refinement |
| `d47ddc1` | drop stale idle before speech | not in any plan |
| `2c871e2` | idle breathing for single-image avatars | S4a-c **done** |
| `4dc2818` | all 8 emotions in the UI | S1 **done** |
| `448ae63` | `_pick_xd` row-consumption test | not in any plan |
| `7b6dc85` | latest-ditto default avatar resolution | not in any plan |

## Next, in value order

1. **T9 — overlap_v2 A/B (65/60).** The lever is built (`DITTO_OVERLAP_V2`,
   default 70) and has never been run. The plan calls it the biggest remaining
   lever: the LMDM pass cost is fixed per window, so a wider `valid_clip_len`
   spreads it over more frames. Quality gate is fuse-seam motion — needs eyes.
2. **T8 — LMDM steps 10→8/6.** Env-only and reversible. ~0.1-0.3 fps of
   contention relief; needs an eyeball A/B.
3. **S3d / S4e — the two box eyeballs.** Register a still image, then watch
   idle→speech→idle. Both implemented; both user-gated.
4. **T7 re-gate.** The todo records 15.4 fps from b14, but b15 and b16 both
   changed the hot path since. One 45 s run at the serving config.
5. **T15 (optional)** — fp16 grid samples in the warp hybrid, ~0.2-0.5 fps,
   gated on the 0.0002 numerics check.

## S5b — answered: repeating an emotion label does nothing

Read from `~/ditto-probe/core/atomic_components/condition_handler.py:6-16`:

```python
def _get_emo_avg(idx=6):
    emo_avg = np.zeros(8, dtype=np.float32)
    if isinstance(idx, (list, tuple)):
        for i in idx:
            emo_avg[i] = 8        # assignment, so duplicates collapse
    else:
        emo_avg[idx] = 8
    emo_avg = softmax(emo_avg)
```

Verified numerically:

| input | happy | neutral |
|---|---|---|
| `[happy]` | 0.997657 | 0.000335 |
| `[happy, happy]` | 0.997657 | 0.000335 |
| `[happy, happy, happy]` | 0.997657 | 0.000335 |
| `[happy, neutral]` | 0.499497 | 0.499497 |
| `[happy, happy, neutral]` | 0.499497 | 0.499497 |

Two consequences that matter for the thesis:

- **Repetition is a no-op**, so intensity cannot come from repeating a label.
  It would need a weighted blend, which the integer-label API cannot express.
- **`softmax(8·one_hot)` is already saturated** at 0.9977. The model sees an
  effectively hard one-hot for single labels and a 50/50 blend for pairs —
  nothing in between. The thesis's valence-arousal experiment (Task 10) must
  build its own soft 8-vector rather than approximating one from label lists.

## AGENTS.md claims — verified 2026-11-19

Checked because the file is the contract other agents work from. Three were
wrong and are now corrected in place:

| Claim | Reality |
|---|---|
| "`app.py` binds `0.0.0.0` today" | `--bind` defaults to `127.0.0.1` (`config.py:110`); `app.py` binds `opt.bind`. Already correct. |
| pytest blocks on live checks without `--ignore` | Every live script lacks a `test_` prefix, so `pytest tests/` never collects them (0 collected, verified). The `--ignore=` flags are no-ops. |
| registry categories `stt, llm, tts, avatar, output` | Only `avatar`, `tts`, `streamout` are registered. Streams are `streamout`; `stt`/`llm` have no registrations. |

The `test_audio_pacing.py` note was also mislabelled "load-flaky". It fails on
macOS from timer/condvar granularity — measured: a bare 50 Hz loop gives 50
ticks, but `Queue.get(timeout=0.01)` on an empty queue takes 32-35 ms. It
reproduces identically on unmodified HEAD. Platform limitation, not load, not a
regression. AGENTS.md now says so.

## Contract violations found and fixed

**`avatars/musetalk_avatar.py` called `.half()` unguarded** (lines 61-63), while
`avatars/musetalk/genavatar.py` guards the same three modules to cuda/mps and
carries a comment explaining that CPU fp16 is emulated in software — "a
generation that should take seconds ran for 10+ minutes without finishing".
The serving path lacked that guard, so a CPU-only host would load MuseTalk in
fp16 and crawl. Now guarded identically. Verified the fp32 path is consistent:
`inference_batch` casts inputs to `unet.model.dtype`.

**`requirements.txt` still declared the dead remote-worker stack.** `grpcio`,
`grpcio-tools`, `protobuf`, `keyring` and `paramiko` existed only for the
local-host + remote-CUDA-box split that AGENTS.md invariant 2 forbids. No module
in the tree imports any of them, and the referenced `docs/remote-gpu.md` does not
exist. Replaced with a comment recording why they must not come back.

## Verified green

`pytest tests/` on the pinned interpreter: **190 passed, 2 failed** — the two
`test_audio_pacing.py` macOS asserts described above. Nothing else regressed.
