import cv2
import numpy as np
import json
from pathlib import Path

_HERE = Path(__file__).parent

VIDEO_PATH      = _HERE.parent / 'video' / 'recording.mkv'
CORNERS_PATH    = _HERE / 'corners.json'
HOMOGRAPHY_PATH = _HERE / 'homography.npy'
OUTPUT_VIDEO    = _HERE.parent / 'video' / (VIDEO_PATH.stem + '-output.mp4')
POSITIONS_PATH  = _HERE / (VIDEO_PATH.stem + '-positions.json')

OUTPUT_WIDTH  = 450
OUTPUT_HEIGHT = 900

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


# Draws the original frame with table outline and detected ball markers,
# and a top-down view with translated ball positions.
def drawFrame(frame, corners, balls, translated):
  # --- Left panel: original with overlays ---
  left = frame.copy()
  pts = corners.astype(np.int32)
  cv2.polylines(left, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
  for cx, cy, r in balls:
    cv2.circle(left, (int(cx), int(cy)), int(r), color=(0, 255, 255), thickness=2)

  # --- Right panel: top-down view ---
  right = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
  cv2.rectangle(right, (0, 0), (OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1),
                color=(255, 255, 255), thickness=2)
  for tx, ty in translated:
    cv2.circle(right, (tx, ty), 6, color=(0, 255, 255), thickness=-1)

  # Resize right panel to match left panel height for side-by-side
  h_left = left.shape[0]
  scale = h_left / right.shape[0]
  right_resized = cv2.resize(right, (int(right.shape[1] * scale), h_left))

  return np.hstack([left, right_resized])


# Processes the video frame by frame: detects balls, translates positions,
# writes an annotated output video, and saves per-frame ball positions to JSON.
def processVideo():
  with open(CORNERS_PATH) as f:
    corners = np.array(json.load(f), dtype=np.float32)
  H = np.load(HOMOGRAPHY_PATH)

  # Precompute table mask
  cap = cv2.VideoCapture(str(VIDEO_PATH))
  assert cap.isOpened(), f"Could not open {VIDEO_PATH}"

  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  table_mask = np.zeros((h, w), dtype=np.uint8)
  cv2.fillPoly(table_mask, [corners.astype(np.int32)], 255)

  # Erode table mask slightly to avoid rail edges
  erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
  table_mask = cv2.erode(table_mask, erode_kernel)

  # Set up output video writer
  sample_frame = np.zeros((h, w, 3), dtype=np.uint8)
  sample_out = drawFrame(sample_frame, corners, [], [])
  out_h, out_w = sample_out.shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (out_w, out_h))

  all_positions = {}
  frame_idx = 0

  print(f"Processing {frame_count} frames at {fps:.0f} fps...")
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    balls = detectBalls(frame, table_mask)
    translated = transformBalls(balls, H)

    # Save positions for this frame
    if translated:
      all_positions[frame_idx] = [(tx, ty) for tx, ty in translated]

    # Write annotated frame
    out_frame = drawFrame(frame, corners, balls, translated)
    writer.write(out_frame)

    if frame_idx % 500 == 0:
      print(f"  Frame {frame_idx}/{frame_count} — {len(balls)} balls detected")

    frame_idx += 1

  cap.release()
  writer.release()

  # Save positions log
  with open(POSITIONS_PATH, 'w') as f:
    json.dump(all_positions, f)

  print(f"Output video: {OUTPUT_VIDEO}")
  print(f"Positions log: {POSITIONS_PATH} ({len(all_positions)} frames with detections)")


if __name__ == '__main__':
  processVideo()
