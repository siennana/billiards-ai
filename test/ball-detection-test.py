import sys
from pathlib import Path
from functools import partial
from ultralytics import YOLO

# Make src/ importable so we can load video_process and detection modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from video_process import processVideo
from detection.balls import detectBallsHoughCircles, detectBallsYOLO, trackBallsYoloTrained

_HERE        = Path(__file__).parent
VIDEO_DIR    = _HERE.parent / 'video'
DATASET_YAML = _HERE.parent / 'datasets' / 'Billiards Detection.yolov8' / 'data.yaml'
WEIGHTS_PATH = _HERE.parent / 'weights' / 'v0' / 'best.pt'


if __name__ == '__main__':
  if len(sys.argv) < 3:
    print("Usage: ball-detection-test.py <input_filename> <output_filename>")
    sys.exit(1)

  video_path   = VIDEO_DIR / sys.argv[1]
  output_stem  = sys.argv[2]

  if not WEIGHTS_PATH.exists():
    print("No trained weights found — training on dataset...")
    model = YOLO('yolov8n.pt')
    model.train(
      data=str(DATASET_YAML),
      epochs=50,
      imgsz=640,
      project=str(DATASET_YAML.parent),
      name='weights',
    )

  print("Running ball detection on video using YOLOv8...")
  model = YOLO(str(WEIGHTS_PATH))
  detect_fn = partial(trackBallsYoloTrained, model=model)
  processVideo(detect_fn, video_path, output_stem, tracePaths=True, trackStats=True)
