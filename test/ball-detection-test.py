import sys
from pathlib import Path

# Make src/ importable so we can load video_process and detection modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from video_process import processVideo
from detection.balls import detectBallsHoughCircles, detectBallsYOLO

_HERE = Path(__file__).parent
VIDEO_PATH = _HERE.parent / 'video' / 'recording-1.mkv'


if __name__ == '__main__':
  # processVideo(detectBallsHoughCircles, VIDEO_PATH, 'recording-1-houghcircles-output')
  processVideo(detectBallsYOLO, VIDEO_PATH, 'recording-1-yolo-output')
