import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

_HERE     = Path(__file__).parent
_REPO     = _HERE.parent
DATASET_DIR = _REPO / 'datasets' / 'billiards'
DATA_YAML   = DATASET_DIR / 'data.yaml'
RUNS_DIR    = _HERE / 'runs'
OUTPUT_DIR  = _REPO / 'weights' / 'v1'

BASE_WEIGHTS = _HERE / 'weights' / 'yolov8n.pt'
VAL_FRACTION = 0.2
SPLIT_SEED   = 42


# The Roboflow export only ships a `train/` split. Carve out a deterministic
# val split on first run so val metrics aren't computed on training data.
def _ensureValSplit():
  train_imgs = DATASET_DIR / 'train' / 'images'
  train_lbls = DATASET_DIR / 'train' / 'labels'
  val_imgs   = DATASET_DIR / 'valid' / 'images'
  val_lbls   = DATASET_DIR / 'valid' / 'labels'

  if val_imgs.exists() and any(val_imgs.iterdir()):
    return

  val_imgs.mkdir(parents=True, exist_ok=True)
  val_lbls.mkdir(parents=True, exist_ok=True)

  images = sorted(p for p in train_imgs.iterdir() if p.is_file())
  rng = random.Random(SPLIT_SEED)
  rng.shuffle(images)
  n_val = max(1, int(len(images) * VAL_FRACTION))
  picked = images[:n_val]

  for img in picked:
    lbl = train_lbls / (img.stem + '.txt')
    shutil.move(str(img), val_imgs / img.name)
    if lbl.exists():
      shutil.move(str(lbl), val_lbls / lbl.name)

  print(f'Created val split: {n_val} of {len(images)} images moved to valid/')


# Ultralytics resolves data.yaml's `path:` against its own configured datasets
# dir (or CWD if `path:` is relative), not against the yaml file's location.
# Rewrite `path:` to an absolute path on each run so training works from any CWD
# and on any machine.
def _writeDataYamlWithAbsolutePath():
  abs_path = str(DATASET_DIR.resolve()).replace('\\', '/')
  DATA_YAML.write_text(
    f'path: {abs_path}\n'
    f'train: train/images\n'
    f'val: valid/images\n'
    f'\n'
    f"nc: 2\n"
    f"names: ['ball', 'chalk']\n"
  )


def main():
  parser = argparse.ArgumentParser(description='Train YOLOv8 on the billiards dataset.')
  parser.add_argument('--epochs',  type=int, default=50)
  parser.add_argument('--imgsz',   type=int, default=640)
  parser.add_argument('--batch',   type=int, default=16)
  parser.add_argument('--device',  default='')         # '' = auto, 'cpu', '0', etc.
  parser.add_argument('--name',    default='billiards')
  parser.add_argument('--weights', default=str(BASE_WEIGHTS),
                      help='Starting checkpoint (defaults to yolo/weights/yolov8n.pt).')
  args = parser.parse_args()

  _ensureValSplit()
  _writeDataYamlWithAbsolutePath()

  model = YOLO(args.weights)
  results = model.train(
    data=str(DATA_YAML),
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    project=str(RUNS_DIR),
    name=args.name,
    exist_ok=False,
  )

  best = Path(results.save_dir) / 'weights' / 'best.pt'
  if best.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / 'best.pt'
    shutil.copy(best, out)
    print(f'Copied best weights to {out}')


if __name__ == '__main__':
  main()
