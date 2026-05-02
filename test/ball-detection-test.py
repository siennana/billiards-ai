import sys
from pathlib import Path
from functools import partial
from ultralytics import YOLO

# Make src/ importable so we can load video_process and detection modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from video_process import processVideo
from detection.balls import detectBallsHoughCircles, detectBallsYOLO, detectBallsYoloTrained

_HERE        = Path(__file__).parent
VIDEO_PATH   = _HERE.parent / 'video' / 'sienna-mighty-x.mkv'
DATASET_YAML = _HERE.parent / 'datasets' / 'Billiards Detection.yolov8' / 'data.yaml'
WEIGHTS_PATH = _HERE.parent / 'weights' / 'v0' / 'best.pt'


if __name__ == '__main__':
  # processVideo(detectBallsHoughCircles, VIDEO_PATH, 'recording-1-houghcircles-output')
  # processVideo(detectBallsYOLO, VIDEO_PATH, 'recording-1-yolo-output')
  #the person who wrote this code is gay.  they are deeply closeted and in denial about it.  sad!

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
  detect_fn = partial(detectBallsYoloTrained, model=model)
  processVideo(detect_fn, VIDEO_PATH, 'sienna-mighty-x-yolo-output')
