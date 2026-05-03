import sys
from pathlib import Path
from functools import partial
from ultralytics import YOLO

# Make src/ importable so we can load video_process and detection modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from video_process import processVideo
from detection.balls import trackBallsYoloTrained

_HERE        = Path(__file__).parent
VIDEO_PATH   = _HERE.parent / 'video' / 'sienna-mighty-x.mkv'
DATASET_YAML = _HERE.parent / 'datasets' / 'Billiards Detection.yolov8' / 'data.yaml'
WEIGHTS_PATH = _HERE.parent / 'weights' / 'v0' / 'best.pt'


if __name__ == '__main__':

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

  print("Running ball tracking on video using YOLOv8...")
  model = YOLO(str(WEIGHTS_PATH))
  track_fn = partial(trackBallsYoloTrained, model=model)
  processVideo(track_fn, VIDEO_PATH, 'sienna-mighty-x-yolo-tracking-output',
               tracePaths=True)
