import cv2
import numpy as np
import json
import os
from pathlib import Path

_HERE = Path(__file__).parent

# dimensions in inches
SHORT_LENGTH = 44
CALIBRATE_IMAGE_PATH = _HERE.parent.parent / 'images' / 'table-snapshot-corners.jpg'
CORNERS_PATH = _HERE / 'corners.json'
HOMOGRAPHY_PATH = _HERE / 'homography.npy'
OUTPUT_WIDTH = 450
OUTPUT_HEIGHT = 900

points = [] # 4 corners of raw table
display_frame = None

# use image with 4 corners marked to identify the raw trapezoidal
# border of the table
def calibrateTable():
    return 0

# use homography to find the birds-eye rectangular table border
def translateTable():
    return 0

# click 4 corners of raw image
# Mouse callback for the calibration window. Records each left-click as a
# corner, draws it on the frame, and connects corners with lines as they accumulate.
def click_event(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
    points.append((x, y))
    print(f"Corner {len(points)}/4: ({x}, {y})")
    cv2.circle(display_frame, (x, y), 6, (0, 0, 255), -1)
    cv2.putText(display_frame, str(len(points)), (x + 8, y - 8),
      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if len(points) > 1:
      cv2.line(display_frame, points[-2], points[-1], (0, 255, 0), 2)
    if len(points) == 4:
      cv2.line(display_frame, points[-1], points[0], (0, 255, 0), 2)
      print("All 4 corners selected. Press any key to save...")
    cv2.imshow("Calibration", display_frame)

# Sorts 4 points into a consistent order — top-left, top-right, bottom-right,
# bottom-left — regardless of the order they were clicked. Uses x+y sum to
# find TL/BR and x-y difference to find TR/BL.
def order_corners(pts):
  pts = np.array(pts, dtype="float32")
  ordered = np.zeros((4, 2), dtype="float32")
  s = pts.sum(axis=1)
  ordered[0] = pts[np.argmin(s)]
  ordered[2] = pts[np.argmax(s)]
  diff = np.diff(pts, axis=1)
  ordered[1] = pts[np.argmin(diff)]
  ordered[3] = pts[np.argmax(diff)]
  return ordered

# Computes the homography matrix that maps the clicked table corners (trapezoid)
# to a flat top-down rectangle of OUTPUT_WIDTH x OUTPUT_HEIGHT. Saves the
# ordered corners to JSON and the matrix to a .npy file for later reuse.
def compute_and_save(corners):
  src = order_corners(corners)
  dst = np.array([
    [0,                0],
    [OUTPUT_WIDTH - 1, 0],
    [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
    [0,                OUTPUT_HEIGHT - 1]
  ], dtype="float32")
  H, _ = cv2.findHomography(src, dst)

  # Save corners as JSON (human readable, easy to inspect)
  with open(CORNERS_PATH, "w") as f:
      json.dump(src.tolist(), f, indent=2)
  print(f"Corners saved to {CORNERS_PATH}")

  # Save homography matrix
  np.save(HOMOGRAPHY_PATH, H)
  print(f"Homography saved to {HOMOGRAPHY_PATH}")

  return H

# Opens the calibration image in a window and waits for the user to click the
# 4 table corners. Once all 4 are selected, computes and saves the homography.
def run_calibration():
  global display_frame
  display_frame = cv2.imread(str(CALIBRATE_IMAGE_PATH)).copy()

  print("Click the 4 table corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
  cv2.imshow("Calibration", display_frame)
  cv2.setMouseCallback("Calibration", click_event)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  assert len(points) == 4, "Need exactly 4 corners"
  H = compute_and_save(points)
  return H

# Loads an existing calibration (corners.json + homography.npy) if present,
# showing a preview so the user can confirm or redo it. Falls back to
# run_calibration() if no saved calibration is found.
def load_or_calibrate():
  if os.path.exists(CORNERS_PATH) and os.path.exists(HOMOGRAPHY_PATH):
    print(f"Found existing calibration in {CORNERS_PATH} — loading...")
    with open(CORNERS_PATH) as f:
        corners = json.load(f)
    print(f"Corners: {corners}")

    # Show the saved corners overlaid on the jpg for a sanity check
    img = cv2.imread(str(CALIBRATE_IMAGE_PATH))
    labels = ["TL", "TR", "BR", "BL"]
    for i, (x, y) in enumerate(corners):
        cv2.circle(img, (int(x), int(y)), 8, (0, 255, 0), -1)
        cv2.putText(img, labels[i], (int(x) + 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Loaded calibration — press R to redo, any other key to continue", img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == ord('r'):
      print("Redoing calibration...")
      return run_calibration()
    else:
      return np.load(HOMOGRAPHY_PATH)
  else:
    print("No calibration found — starting fresh...")
    return run_calibration()

# --- Entry point ---
H = load_or_calibrate()
print("Ready to process video with H:\n", H)