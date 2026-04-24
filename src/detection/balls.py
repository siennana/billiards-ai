import cv2
import numpy as np

# HoughCircles tuning for billiard balls — may need adjusting per camera/resolution
HOUGH_DP         = 1.2
HOUGH_MIN_DIST   = 20
HOUGH_PARAM1     = 120
HOUGH_PARAM2     = 20
HOUGH_MIN_RADIUS = 8
HOUGH_MAX_RADIUS = 20


# Detects billiard balls using the Hough Circle Transform on a grayscale,
# median-blurred frame. Keeps only circles whose centers fall inside table_mask.
# Returns a list of (cx, cy, radius) tuples matching the detector interface.
def detectBallsHoughCircles(frame, table_mask):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  blurred = cv2.medianBlur(gray, 5)

  circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=HOUGH_DP,
    minDist=HOUGH_MIN_DIST,
    param1=HOUGH_PARAM1,
    param2=HOUGH_PARAM2,
    minRadius=HOUGH_MIN_RADIUS,
    maxRadius=HOUGH_MAX_RADIUS,
  )

  if circles is None:
    return []

  h, w = table_mask.shape[:2]
  balls = []
  for x, y, r in circles[0]:
    ix, iy = int(x), int(y)
    if 0 <= iy < h and 0 <= ix < w and table_mask[iy, ix] > 0:
      balls.append((float(x), float(y), float(r)))

  return balls
