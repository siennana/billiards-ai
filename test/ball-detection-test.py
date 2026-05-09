import sys
from pathlib import Path
from functools import partial
from ultralytics import YOLO

# Make src/ importable so we can load video_process and detection modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from video_process import processVideo
from detection.balls import detectBallsHoughCircles, detectBallsYOLO, trackBallsYoloTrained, TRACKER_YAML

_HERE        = Path(__file__).parent
VIDEO_DIR    = _HERE.parent / 'video'
DATASET_YAML = _HERE.parent / 'datasets' / 'Billiards Detection.yolov8' / 'data.yaml'
WEIGHTS_PATH = _HERE.parent / 'weights' / 'v2' / 'best.pt'


if __name__ == '__main__':
  if len(sys.argv) < 3:
    print("Usage: ball-detection-test.py <inputFile> <outputExtension>")
    print("  e.g. ball-detection-test.py ben-alex-clip.mkv yolo-v2")
    print("  -> video/test-output/ben-alex-clip_yolo-v2/{recording.mkv,positions.json,events.json}")
    sys.exit(1)

  inputFile       = sys.argv[1]
  outputExtension = sys.argv[2]
  video_path      = VIDEO_DIR / inputFile
  output_subdir   = f'{Path(inputFile).stem}_{outputExtension}'

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
  detect_fn = partial(trackBallsYoloTrained, model=model, applyFiltering=False)
  processVideo(detect_fn, video_path, output_subdir, tracePaths=True, trackStats=True,
               tracker_yaml=str(TRACKER_YAML), weights=str(WEIGHTS_PATH))
