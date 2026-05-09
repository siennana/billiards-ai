import sys
import json
from pathlib import Path
from functools import partial
import yaml
from ultralytics import YOLO

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from video_process import processVideo
from detection.balls import trackBallsYoloTrained

_HERE        = Path(__file__).parent
WEIGHTS_PATH = _HERE.parent / 'weights' / 'v2' / 'best.pt'
INPUT_CLIP   = _HERE.parent / 'video' / 'ben-sienna-20260503_232606-cfr-clip.mp4'
OUTPUT_BASE  = _HERE.parent / 'video' / 'test-output' / 'botsort-tunings'

# Mirrors yolo/trackers/botsort.yaml — the starting point each sweep
# overrides one parameter against. Keep in sync with that file if you
# change defaults you actually use in production.
BASELINE = {
  'tracker_type':      'botsort',
  'track_high_thresh': 0.4,
  'track_low_thresh':  0.1,
  'new_track_thresh':  0.6,
  'track_buffer':      300,
  'match_thresh':      0.85,
  'fuse_score':        True,
  'gmc_method':        'sparseOptFlow',
  'proximity_thresh':  0.5,
  'appearance_thresh': 0.25,
  'with_reid':         False,
}

# Each entry overrides exactly one BASELINE field. 20 runs total — one
# parameter swept per range. Lets you isolate the effect of each knob.
SWEEP = [
  # track_buffer (6 runs) — frames a lost track is kept alive on Kalman
  ('track_buffer',      30),
  ('track_buffer',      90),
  ('track_buffer',      150),
  ('track_buffer',      300),   # baseline
  ('track_buffer',      500),
  ('track_buffer',      1000),

  # match_thresh (5 runs) — max cost (1 - IoU) accepted for a match
  ('match_thresh',      0.7),
  ('match_thresh',      0.8),
  ('match_thresh',      0.85),  # baseline
  ('match_thresh',      0.9),
  ('match_thresh',      0.95),

  # new_track_thresh (5 runs) — min conf to spawn a new track
  ('new_track_thresh',  0.4),
  ('new_track_thresh',  0.5),
  ('new_track_thresh',  0.6),   # baseline
  ('new_track_thresh',  0.7),
  ('new_track_thresh',  0.8),

  # track_high_thresh (4 runs) — first-pass match conf threshold
  ('track_high_thresh', 0.3),
  ('track_high_thresh', 0.4),   # baseline
  ('track_high_thresh', 0.5),
  ('track_high_thresh', 0.6),
]


def _summarize(positions_path):
  """Read positions.json and return (max_id, unique_count, frames_with_balls)."""
  with open(positions_path) as f:
    positions = json.load(f)
  ids = set()
  for entries in positions.values():
    for entry in entries:
      ids.add(entry[2])  # entry shape: [tx, ty, ball_id, conf]
  max_id = max(ids) if ids else 0
  return max_id, len(ids), len(positions)


if __name__ == '__main__':
  if not INPUT_CLIP.exists():
    sys.exit(f"Test clip not found: {INPUT_CLIP}")
  if not WEIGHTS_PATH.exists():
    sys.exit(f"Weights not found: {WEIGHTS_PATH}")

  OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
  print(f"Sweeping {len(SWEEP)} configs on {INPUT_CLIP.name} -> {OUTPUT_BASE}")

  results = []
  for i, (param, value) in enumerate(SWEEP, start=1):
    run_dir = OUTPUT_BASE / str(i)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = dict(BASELINE)
    config[param] = value
    yaml_path = run_dir / 'botsort.yaml'
    with open(yaml_path, 'w') as f:
      yaml.safe_dump(config, f, sort_keys=False)

    print(f"\n=== Run {i}/{len(SWEEP)}: {param}={value} -> {run_dir} ===")

    # Reload the model each run so BoT-SORT's persistent tracker state
    # from the previous run doesn't leak into this one.
    model = YOLO(str(WEIGHTS_PATH))
    detect_fn = partial(trackBallsYoloTrained,
                        model=model,
                        applyFiltering=False,
                        tracker_yaml=str(yaml_path))

    # output_path is relative to OUTPUT_DIR (video/test-output/)
    rel_output = f'botsort-tunings/{i}'
    processVideo(detect_fn, INPUT_CLIP, rel_output,
                 tracePaths=True, trackStats=True)

    positions_path = run_dir / 'positions.json'
    if positions_path.exists():
      max_id, unique_count, frames = _summarize(positions_path)
      results.append((i, param, value, max_id, unique_count, frames))
      print(f"  -> max_id={max_id}, unique_ids={unique_count}, frames_with_balls={frames}")

  print("\n=== Summary ===")
  print(f"{'#':>3}  {'param':<20} {'value':>10}  {'max_id':>8}  {'unique':>8}  {'frames':>8}")
  for i, param, value, max_id, unique_count, frames in results:
    print(f"{i:>3}  {param:<20} {str(value):>10}  {max_id:>8}  {unique_count:>8}  {frames:>8}")
