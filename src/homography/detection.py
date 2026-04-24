import cv2
import numpy as np

# HSV range for the blue table felt
FELT_LOWER = np.array([85, 20, 80])
FELT_UPPER = np.array([135, 180, 220])

# Ball detection filters
MIN_AREA = 80
MAX_AREA = 800
MIN_CIRCULARITY = 0.5


# Detects billiard balls on a single frame by finding non-felt-colored blobs
# within the table region that are circular and ball-sized.
def detectBalls(frame, table_mask):
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  felt_mask = cv2.inRange(hsv, FELT_LOWER, FELT_UPPER)

  # Non-felt pixels within the table area = ball candidates
  candidates = cv2.bitwise_and(cv2.bitwise_not(felt_mask), table_mask)

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)
  candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, kernel)

  contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  balls = []
  for c in contours:
    area = cv2.contourArea(c)
    if area < MIN_AREA or area > MAX_AREA:
      continue
    perim = cv2.arcLength(c, True)
    if perim == 0:
      continue
    circularity = 4 * np.pi * area / (perim ** 2)
    if circularity < MIN_CIRCULARITY:
      continue
    (cx, cy), radius = cv2.minEnclosingCircle(c)
    balls.append((float(cx), float(cy), float(radius)))

  return balls


# Applies the homography matrix to a list of (cx, cy, r) ball positions,
# returning translated (tx, ty) points in top-down coordinates.
def transformBalls(balls, H):
  if not balls:
    return []
  src = np.array([[cx, cy] for cx, cy, _ in balls], dtype=np.float32).reshape(-1, 1, 2)
  dst = cv2.perspectiveTransform(src, H)
  return [(int(p[0][0]), int(p[0][1])) for p in dst]
