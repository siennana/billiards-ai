import sys
import json
from pathlib import Path
import cv2
import numpy as np

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from detection.balls import detectBallsHoughCircles

_HERE        = Path(__file__).parent
IMAGE_PATH   = _HERE.parent / 'images' / 'table-snapshot-back-rail.jpg'
CORNERS_PATH = _HERE.parent / 'src' / 'homography' / 'corners.json'
OUTPUT_PATH  = _HERE.parent / 'images' / 'test-output' / 'table-snapshot-back-rail-houghcircles.jpg'


if __name__ == '__main__':
  img = cv2.imread(str(IMAGE_PATH))
  assert img is not None, f"Could not read {IMAGE_PATH}"

  with open(CORNERS_PATH) as f:
    corners = np.array(json.load(f), dtype=np.float32)

  h, w = img.shape[:2]
  table_mask = np.zeros((h, w), dtype=np.uint8)
  cv2.fillPoly(table_mask, [corners.astype(np.int32)], 255)

  # Erode mask to avoid rail edges (mirrors the video pipeline)
  erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
  table_mask = cv2.erode(table_mask, erode_kernel)

  balls = detectBallsHoughCircles(img, table_mask)
  print(f"Balls detected: {len(balls)}")

  out = img.copy()
  cv2.polylines(out, [corners.astype(np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)
  for cx, cy, r in balls:
    cv2.circle(out, (int(cx), int(cy)), int(r), color=(0, 255, 255), thickness=2)

  cv2.imwrite(str(OUTPUT_PATH), out)
  print(f"Saved: {OUTPUT_PATH}")
