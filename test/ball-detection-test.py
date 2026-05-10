import argparse
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
WEIGHTS_PATH = _HERE.parent / 'weights' / 'v5' / 'best.pt'


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Run YOLO ball detection / BoT-SORT tracking on a video.',
    epilog='Output -> video/test-output/<inputStem>_<outputExtension>/')
  parser.add_argument('inputFile',       help='video filename in video/ (with extension)')
  parser.add_argument('outputExtension', help='label suffix for the output subdir')
  parser.add_argument('--mode', choices=['tracking', 'detection', 'both'], default='both',
                      help='tracking = BoT-SORT IDs + tracked-recording.mkv + positions/events/metadata; '
                           'detection = raw YOLO + detected-recording.mkv + detected-positions; '
                           'both (default) = everything (~2x inference time)')
  args = parser.parse_args()

  video_path    = VIDEO_DIR / args.inputFile
  output_subdir = f'{Path(args.inputFile).stem}_{args.outputExtension}'

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

  enableTracking  = args.mode in ('tracking', 'both')
  enableDetection = args.mode in ('detection', 'both')

  print(f"Running ball detection on video using YOLOv8 (mode={args.mode})...")
  model = YOLO(str(WEIGHTS_PATH))
  detect_fn = partial(trackBallsYoloTrained, model=model, applyFiltering=False)
  processVideo(detect_fn, video_path, output_subdir, tracePaths=True, trackStats=True,
               tracker_yaml=str(TRACKER_YAML), weights=str(WEIGHTS_PATH),
               enableTracking=enableTracking, enableDetection=enableDetection)
