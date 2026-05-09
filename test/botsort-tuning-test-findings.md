# BoT-SORT Tuning Sweep — Findings

Results from running [test/botsort-tuning-test.py](botsort-tuning-test.py) — a 20-run parameter sweep over BoT-SORT's main tuning knobs, applied to the `ben-sienna-20260503_232606-cfr-clip.mp4` test clip (5100 frames, 60 fps, 1088×1456).

## Setup

- **Model**: `weights/v2/best.pt` (3-class trained YOLOv8n: ball, chalk, cue stick)
- **Detection**: `trackBallsYoloTrained` with `conf=CONFIDENCE_LOW=0.4`, `applyFiltering=False`
- **Baseline config** (mirrors [yolo/trackers/botsort.yaml](../yolo/trackers/botsort.yaml) at sweep time):
  - `track_high_thresh: 0.4`
  - `track_low_thresh:  0.1`
  - `new_track_thresh:  0.6`
  - `track_buffer:      300`
  - `match_thresh:      0.85`
  - `fuse_score:        True`
  - `gmc_method:        sparseOptFlow`
- **Sweep strategy**: vary one parameter at a time over a useful range, 20 runs total.
- **Metrics**:
  - `max_id` — highest ID assigned by BoT-SORT (includes spurious one-frame tracks)
  - `unique_ids` — distinct IDs that survived ≥1 frame
  - `frames` — frames with ≥1 detection (sanity check; should be 5100)

Lower `unique_ids` = more stable tracking. The clip has 16 physical balls, so 16 is the unreachable theoretical floor.

## Full results

```
 #   param               value     max_id   unique_ids   frames
─── track_buffer ────────────────────────────────────────────────
 1   track_buffer            30       163          109     5100
 2   track_buffer            90       155          106     5100
 3   track_buffer           150       153          105     5100
 4   track_buffer           300       153          105     5100   ← baseline
 5   track_buffer           500       153          105     5100
 6   track_buffer          1000       152          105     5100
─── match_thresh ────────────────────────────────────────────────
 7   match_thresh           0.7       286          198     5100
 8   match_thresh           0.8       189          124     5100
 9   match_thresh          0.85       153          105     5100   ← baseline
10   match_thresh           0.9       136           87     5100   ← winner
11   match_thresh          0.95       103           68     5100   ← careful
─── new_track_thresh ────────────────────────────────────────────
12   new_track_thresh       0.4       210          108     5100
13   new_track_thresh       0.5       175          105     5100
14   new_track_thresh       0.6       153          105     5100   ← baseline
15   new_track_thresh       0.7       137          103     5100
16   new_track_thresh       0.8       123          101     5100
─── track_high_thresh ───────────────────────────────────────────
17   track_high_thresh      0.3       153          105     5100
18   track_high_thresh      0.4       153          105     5100   ← baseline
19   track_high_thresh      0.5       156          105     5100
20   track_high_thresh      0.6       159          105     5100
```

## Findings

### 1. `match_thresh` is the dominant lever
Going from `0.85` to `0.90` cut unique IDs from **105 → 87** (~17% drop). Pushing further to `0.95` cut to **68** (~35% from baseline). This single parameter has more impact on ID stability than every other knob combined.

**Why**: `match_thresh` is the maximum cost (`1 - IoU`) accepted when assigning a new detection to an existing track. Raising it = looser IoU matching, which lets BoT-SORT reattach a slightly-shifted detection to the original track instead of spawning a fresh ID.

**Caveat**: this metric (unique IDs) doesn't catch the failure mode of `match_thresh` being *too* loose — incorrectly merging two physically different balls into one track. `0.95` looks great by the numbers but should be treated with suspicion until visually validated. `0.90` is the safer choice and was committed to production.

### 2. `track_buffer` plateaus past ~150
Bumping from `30 → 90` saved 3 unique IDs. `90 → 150` saved 1 more. **`150 → 300 → 500 → 1000` saved zero more.**

Translation: the kind of dropouts `track_buffer` rescues you from (Kalman coasting through occlusions ≤5 sec at 30fps) are a relatively small contributor on this clip. The remaining ~105 IDs come from failures `track_buffer` can't address (YOLO outright missing a ball mid-frame for visual reasons).

**Action**: dropped `track_buffer` from `300 → 150`. Same result, less memory.

### 3. `new_track_thresh` is mostly cosmetic
Raising it (0.6 → 0.7 → 0.8) reduced `max_id` significantly (153 → 137 → 123) but barely moved `unique_ids` (105 → 103 → 101).

The gap between `max_id` and `unique_ids` is **spurious one-frame tracks** — flicker that BoT-SORT briefly assigns an ID to before discarding. Higher `new_track_thresh` suppresses this noise but doesn't help with real long-lived ID fragmentation.

**Action**: raised to `0.7` — minor improvement, no downside.

### 4. `track_high_thresh` barely matters here
Variance across `0.3 → 0.6` was within noise (153 → 159). Likely because `conf=CONFIDENCE_LOW=0.4` already filters detections at the YOLO inference call before the tracker ever sees them, so this second-pass threshold has limited material to work with.

**Action**: left unchanged. (Subsequently set to `0.6` manually for other reasons unrelated to this sweep.)

## Final config committed

Applied to [yolo/trackers/botsort.yaml](../yolo/trackers/botsort.yaml):

```yaml
track_high_thresh: 0.6     # later manual edit, unrelated to sweep
track_low_thresh:  0.35    # later manual edit, unrelated to sweep
new_track_thresh: 0.7      # was 0.6 — finding #3
track_buffer:     150      # was 300 — finding #2
match_thresh:     0.9      # was 0.85 — finding #1
gmc_method:       none     # later manual edit, unrelated to sweep
```

## The bigger picture: tracker tuning has a ceiling

Best result with online tracking alone: **~68 unique IDs** (at `match_thresh=0.95`, with risk of false merges).

For a 16-ball table, that's still **4× the theoretical floor**. The remaining gap is from causes that are structurally outside what BoT-SORT can fix:

- YOLO occasionally drops a detection for several frames (lighting, motion blur, partial occlusion by another ball)
- BoT-SORT's Kalman gives up after `track_buffer` and can't be told "this ball was stationary, look longer"
- No visual ReID — even if there were, billiard balls (solid colors) defeat appearance-based ReID

**Closing the gap further requires offline track stitching** — a post-process that operates on the saved trajectory data with global view, looking forward through fragmented tracks and merging them based on motion prediction + spatial proximity. That work is tracked separately and will live in `src/track_stitching.py`.

## Per-run artifacts

Each of the 20 runs has its own subdirectory under `video/test-output/botsort-tunings/<N>/`:

- `botsort.yaml` — exact config used for that run
- `recording.mkv` — annotated side-by-side video
- `positions.json` — per-frame ball positions with IDs and confidence
- `events.json` — pocket/shot/rack events derived from positions

Re-tune any threshold offline by re-running [src/stat_tracking.py](../src/stat_tracking.py) on a run's `positions.json` — no need to rerun YOLO.

## Reproducing

```bash
python test/botsort-tuning-test.py
```

Edit the `SWEEP` list at the top of [test/botsort-tuning-test.py](botsort-tuning-test.py) to test different parameter ranges. Each new run number gets its own subdir; nothing from previous sweeps is overwritten unless you reuse run numbers.
