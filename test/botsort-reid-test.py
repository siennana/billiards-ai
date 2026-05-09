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
OUTPUT_BASE  = _HERE.parent / 'video' / 'test-output' / 'botsort-reid-test'

# Production config from yolo/trackers/botsort.yaml. The two runs differ
# only in with_reid so any delta is attributable to ReID alone.
BASELINE = {
  'tracker_type':      'botsort',
  'track_high_thresh': 0.6,
  'track_low_thresh':  0.35,
  'new_track_thresh':  0.7,
  'track_buffer':      150,
  'match_thresh':      0.9,
  'fuse_score':        True,
  'gmc_method':        'none',
  'proximity_thresh':  0.5,
  'appearance_thresh': 0.25,
  'model':             'auto',
}

RUNS = [
  ('with_reid_false', {'with_reid': False}),
  ('with_reid_true',  {'with_reid': True}),
]


def _summarize(positions_path):
  with open(positions_path) as f:
    positions = json.load(f)
  ids = set()
  for entries in positions.values():
    for entry in entries:
      ids.add(entry[2])
  return (max(ids) if ids else 0), len(ids), len(positions)


if __name__ == '__main__':
  if not INPUT_CLIP.exists():
    sys.exit(f"Test clip not found: {INPUT_CLIP}")
  if not WEIGHTS_PATH.exists():
    sys.exit(f"Weights not found: {WEIGHTS_PATH}")

  OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
  print(f"Comparing {len(RUNS)} configs on {INPUT_CLIP.name} -> {OUTPUT_BASE}")

  results = []
  for name, override in RUNS:
    run_dir = OUTPUT_BASE / name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = dict(BASELINE)
    config.update(override)
    yaml_path = run_dir / 'botsort.yaml'
    with open(yaml_path, 'w') as f:
      yaml.safe_dump(config, f, sort_keys=False)

    print(f"\n=== {name}: with_reid={config['with_reid']} -> {run_dir} ===")

    model = YOLO(str(WEIGHTS_PATH))
    detect_fn = partial(trackBallsYoloTrained,
                        model=model,
                        applyFiltering=False,
                        tracker_yaml=str(yaml_path))

    rel_output = f'botsort-reid-test/{name}'
    processVideo(detect_fn, INPUT_CLIP, rel_output,
                 tracePaths=True, trackStats=True,
                 tracker_yaml=str(yaml_path), weights=str(WEIGHTS_PATH))

    positions_path = run_dir / 'positions.json'
    if positions_path.exists():
      max_id, unique, frames = _summarize(positions_path)
      results.append((name, config['with_reid'], max_id, unique, frames))
      print(f"  -> max_id={max_id}, unique_ids={unique}, frames_with_balls={frames}")

  print("\n=== Summary ===")
  print(f"{'name':<20} {'with_reid':>10} {'max_id':>8} {'unique':>8} {'frames':>8}")
  for name, with_reid, max_id, unique, frames in results:
    print(f"{name:<20} {str(with_reid):>10} {max_id:>8} {unique:>8} {frames:>8}")
