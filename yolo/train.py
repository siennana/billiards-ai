import argparse
import fnmatch
import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

_HERE     = Path(__file__).parent
_REPO     = _HERE.parent
DATASET_DIR = _REPO / 'datasets' / 'billiards-5'
DATA_YAML   = DATASET_DIR / 'data.yaml'
SPLITS_JSON = DATASET_DIR / 'splits.json'
RUNS_DIR    = _HERE / 'runs'
OUTPUT_DIR  = _REPO / 'weights' / 'v5'

# Default fine-tunes from v4. For from-scratch training pass --weights yolo/weights/yolov8n.pt.
BASE_WEIGHTS = _REPO / 'weights' / 'v4' / 'best.pt'
SPLIT_SEED   = 42
DEFAULT_RATIOS = {'train': 0.8, 'valid': 0.2, 'test': 0.0}


# Roboflow exports ship a `train/` only. Carve the val (and optionally test)
# split deterministically. If a sidecar `splits.json` exists in the dataset
# dir, honor explicit per-image patterns first; otherwise do a flat 80/20
# train/valid split.
#
# splits.json schema (all keys optional, sensible defaults applied):
#   {
#     "valid_patterns": ["frame_18_*", ...],   # fnmatch globs against filename
#     "test_patterns":  ["frame_*"],
#     "train_patterns": ["frame_*"],           # force-route to train (skips ratio split)
#     "default_ratios": {"train": 0.8, "valid": 0.2, "test": 0.0}
#   }
#
# Precedence: valid_patterns > test_patterns > train_patterns > default_ratios.
# Splits are sticky once created — pass force=True to wipe valid/ and test/
# back to train/ first. Useful when iterating on splits.json.
def _ensureSplits(force=False):
  train_imgs = DATASET_DIR / 'train' / 'images'
  train_lbls = DATASET_DIR / 'train' / 'labels'
  val_imgs   = DATASET_DIR / 'valid' / 'images'
  val_lbls   = DATASET_DIR / 'valid' / 'labels'
  test_imgs  = DATASET_DIR / 'test'  / 'images'
  test_lbls  = DATASET_DIR / 'test'  / 'labels'

  if force:
    for src_imgs, src_lbls in [(val_imgs, val_lbls), (test_imgs, test_lbls)]:
      if src_imgs.exists():
        for img in list(src_imgs.iterdir()):
          shutil.move(str(img), train_imgs / img.name)
      if src_lbls.exists():
        for lbl in list(src_lbls.iterdir()):
          shutil.move(str(lbl), train_lbls / lbl.name)

  if val_imgs.exists() and any(val_imgs.iterdir()):
    return

  cfg = None
  if SPLITS_JSON.exists():
    raw = json.loads(SPLITS_JSON.read_text())
    cfg = (
      raw.get('valid_patterns', []),
      raw.get('test_patterns',  []),
      raw.get('train_patterns', []),
      {**DEFAULT_RATIOS, **raw.get('default_ratios', {})},
    )

  images = sorted(p for p in train_imgs.iterdir() if p.is_file())
  val_imgs.mkdir(parents=True, exist_ok=True)
  val_lbls.mkdir(parents=True, exist_ok=True)

  if cfg is None:
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * DEFAULT_RATIOS['valid']))
    for img in images[:n_val]:
      _moveImageAndLabel(img, train_lbls, val_imgs, val_lbls)
    print(f'Created val split: {n_val} of {len(images)} images moved to valid/')
    return

  test_imgs.mkdir(parents=True, exist_ok=True)
  test_lbls.mkdir(parents=True, exist_ok=True)

  valid_patterns, test_patterns, train_patterns, ratios = cfg
  explicit_valid, explicit_test, remaining = [], [], []
  forced_train_count = 0
  for img in images:
    if any(fnmatch.fnmatch(img.name, p) for p in valid_patterns):
      explicit_valid.append(img)
    elif any(fnmatch.fnmatch(img.name, p) for p in test_patterns):
      explicit_test.append(img)
    elif any(fnmatch.fnmatch(img.name, p) for p in train_patterns):
      forced_train_count += 1   # already in train/, no move needed
    else:
      remaining.append(img)

  rng = random.Random(SPLIT_SEED)
  rng.shuffle(remaining)
  n_val_extra  = int(len(remaining) * ratios['valid'])
  n_test_extra = int(len(remaining) * ratios['test'])
  picked_val_extra  = remaining[:n_val_extra]
  picked_test_extra = remaining[n_val_extra:n_val_extra + n_test_extra]

  for img in explicit_valid + picked_val_extra:
    _moveImageAndLabel(img, train_lbls, val_imgs, val_lbls)
  for img in explicit_test + picked_test_extra:
    _moveImageAndLabel(img, train_lbls, test_imgs, test_lbls)

  total_val  = len(explicit_valid) + len(picked_val_extra)
  total_test = len(explicit_test)  + len(picked_test_extra)
  print(f'Splits: valid={total_val} (explicit={len(explicit_valid)}), '
        f'test={total_test} (explicit={len(explicit_test)}), '
        f'train={len(images) - total_val - total_test} (forced={forced_train_count})')


def _moveImageAndLabel(img_path, src_lbl_dir, dst_imgs, dst_lbls):
  shutil.move(str(img_path), dst_imgs / img_path.name)
  lbl = src_lbl_dir / (img_path.stem + '.txt')
  if lbl.exists():
    shutil.move(str(lbl), dst_lbls / lbl.name)


# Ultralytics resolves data.yaml's `path:` against its own configured datasets
# dir (or CWD if `path:` is relative), not against the yaml file's location.
# Rewrite `path:` to an absolute path on each run so training works from any CWD
# and on any machine.
def _writeDataYamlWithAbsolutePath():
  abs_path = str(DATASET_DIR.resolve()).replace('\\', '/')
  test_dir = DATASET_DIR / 'test' / 'images'
  test_line = 'test: test/images\n' if test_dir.exists() else ''
  DATA_YAML.write_text(
    f'path: {abs_path}\n'
    f'train: train/images\n'
    f'val: valid/images\n'
    f'{test_line}'
    f'\n'
    f"nc: 3\n"
    f"names: ['ball', 'chalk', 'cue stick']\n"
  )


def main():
  parser = argparse.ArgumentParser(description='Train YOLOv8 on the billiards dataset.')
  parser.add_argument('--epochs',   type=int, default=50)
  parser.add_argument('--imgsz',    type=int, default=960)
  parser.add_argument('--batch',    type=int, default=8)
  parser.add_argument('--workers',  type=int, default=2)  # Windows: keep low to avoid pagefile OOM
  parser.add_argument('--device',   default='')         # '' = auto, 'cpu', '0', etc.
  parser.add_argument('--name',     default='billiards-5')
  parser.add_argument('--patience', type=int, default=20)
  parser.add_argument('--weights',  default=str(BASE_WEIGHTS),
                      help='Starting checkpoint (defaults to weights/v4/best.pt).')
  parser.add_argument('--resplit',  action='store_true',
                      help='Wipe valid/ and test/ back to train/ before recomputing splits.')
  args = parser.parse_args()

  _ensureSplits(force=args.resplit)
  _writeDataYamlWithAbsolutePath()

  model = YOLO(args.weights)
  results = model.train(
    data=str(DATA_YAML),
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    workers=args.workers,
    device=args.device,
    patience=args.patience,
    project=str(RUNS_DIR),
    name=args.name,
    exist_ok=False,
    # Fine-tuning from v4 with a low LR keeps existing knowledge intact
    # while letting newly-added images nudge the weights. Strong
    # augmentation multiplies each rare image's effective contribution.
    lr0=0.001,
    mixup=0.1,
    degrees=10,
    scale=0.5,
    translate=0.1,
    cos_lr=True,
  )

  best = Path(results.save_dir) / 'weights' / 'best.pt'
  if best.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / 'best.pt'
    shutil.copy(best, out)
    print(f'Copied best weights to {out}')


if __name__ == '__main__':
  main()
