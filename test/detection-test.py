import cv2
import numpy as np
import json
from pathlib import Path

_HERE = Path(__file__).parent

IMAGE_PATH      = _HERE.parent / 'images' / 'table-snapshot-balls.jpg'
CORNERS_PATH    = _HERE.parent / 'src' / 'homography' / 'corners.json'
HOMOGRAPHY_PATH = _HERE.parent / 'src' / 'homography' / 'homography.npy'
PRE_OUTPUT_PATH = _HERE.parent / 'images' / 'pre-homography-overlay.jpg'
POST_OUTPUT_PATH = _HERE.parent / 'images' / 'post-homography-detection-output.jpg'

OUTPUT_WIDTH  = 450
OUTPUT_HEIGHT = 900

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICK = 2

# Detects the red-circled billiard balls in the input image, filters to those
# inside the saved table corners, applies the homography to translate each center
# to the top-down view, and saves two output images: a pre-homography overlay
# and a post-homography top-down view.
def testDetection():
  img = cv2.imread(str(IMAGE_PATH))
  assert img is not None, f"Could not read {IMAGE_PATH}"

  # Load calibration data
  with open(CORNERS_PATH) as f:
    corners = np.array(json.load(f), dtype=np.float32)
  H = np.load(HOMOGRAPHY_PATH)

  # --- Detect red circles ---
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  # Red wraps around the HSV hue axis, so threshold both ends
  red_lo = cv2.inRange(hsv, np.array([0,   120, 50]), np.array([10,  255, 255]))
  red_hi = cv2.inRange(hsv, np.array([160, 120, 50]), np.array([180, 255, 255]))
  red_mask = cv2.bitwise_or(red_lo, red_hi)

  # Close small gaps in the painted circles
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
  red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

  contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Build the table boundary polygon for inside-check
  table_poly = corners.astype(np.int32).reshape(-1, 1, 2)

  # Find the center of each red contour inside the table
  ball_centers = []
  for c in contours:
    if cv2.contourArea(c) < 200:  # discard specks
      continue
    (cx, cy), _ = cv2.minEnclosingCircle(c)
    if cv2.pointPolygonTest(table_poly, (float(cx), float(cy)), False) >= 0:
      ball_centers.append((float(cx), float(cy)))

  print(f"Balls detected: {len(ball_centers)}")

  # --- Pre-homography overlay ---
  pre = img.copy()

  # Draw trapezoid outline and labeled corners in white
  pts = corners.astype(np.int32)
  for i in range(4):
    cv2.line(pre, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color=(255, 255, 255), thickness=2)
  for i, (x, y) in enumerate(pts):
    cv2.circle(pre, (x, y), 6, color=(255, 255, 255), thickness=-1)
    cv2.putText(pre, str(i + 1), (x + 8, y - 8), FONT, FONT_SCALE, (255, 255, 255), FONT_THICK)

  # Draw yellow dot at each ball center
  for cx, cy in ball_centers:
    cv2.circle(pre, (int(cx), int(cy)), 5, color=(0, 255, 255), thickness=-1)

  cv2.imwrite(str(PRE_OUTPUT_PATH), pre)
  print(f"Saved: {PRE_OUTPUT_PATH}")

  # --- Apply homography to translate ball centers ---
  translated = []
  if ball_centers:
    src_pts = np.array(ball_centers, dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = cv2.perspectiveTransform(src_pts, H)
    translated = [(int(p[0][0]), int(p[0][1])) for p in dst_pts]

  # Also translate the 4 corners to confirm their mapped positions
  corner_src = corners.reshape(-1, 1, 2)
  corner_dst = cv2.perspectiveTransform(corner_src, H)
  corner_dst = [(int(p[0][0]), int(p[0][1])) for p in corner_dst]

  # --- Post-homography top-down view ---
  post = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)

  # Table rectangle
  cv2.rectangle(post, (0, 0), (OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1),
                color=(255, 255, 255), thickness=2)

  # Labeled corners
  for i, (x, y) in enumerate(corner_dst):
    cv2.circle(post, (x, y), 5, color=(255, 255, 255), thickness=-1)
    cv2.putText(post, str(i + 1), (x + 8, y - 8), FONT, FONT_SCALE, (255, 255, 255), FONT_THICK)

  # Yellow dot for each translated ball
  for tx, ty in translated:
    cv2.circle(post, (tx, ty), 5, color=(0, 255, 255), thickness=-1)

  cv2.imwrite(str(POST_OUTPUT_PATH), post)
  print(f"Saved: {POST_OUTPUT_PATH}")


if __name__ == '__main__':
  testDetection()
