import cv2
import numpy as np


# Applies the homography matrix to a list of (cx, cy, r) ball positions,
# returning translated (tx, ty) points in top-down coordinates.
def transformBalls(balls, H):
  if not balls:
    return []
  src = np.array([[cx, cy] for cx, cy, _ in balls], dtype=np.float32).reshape(-1, 1, 2)
  dst = cv2.perspectiveTransform(src, H)
  return [(int(p[0][0]), int(p[0][1])) for p in dst]
