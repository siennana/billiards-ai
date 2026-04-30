import cv2
import numpy as np
import json
from pathlib import Path

from homography.detection import transformBalls

_HERE = Path(__file__).parent

VIDEO_PATH      = _HERE.parent / 'video' / 'recording.mkv'
VIDEO_PATH_1      = _HERE.parent / 'video' / 'recording-1.mkv'
CORNERS_PATH    = _HERE / 'homography' / 'corners.json'
HOMOGRAPHY_PATH = _HERE / 'homography' / 'homography.npy'
OUTPUT_DIR      = _HERE.parent / 'video' / 'test-output'

OUTPUT_WIDTH  = 450
OUTPUT_HEIGHT = 900


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


# Processes the video frame by frame using the provided detect_fn, translates
# ball positions to top-down coords, writes an annotated output video, and
# saves per-frame ball positions to JSON. Both outputs are named using
# output_name and placed in video/test-output.
#
# detect_fn signature: (frame, table_mask) -> list[(cx, cy, r)]
def processVideo(detect_fn, input_path, output_path):
  output_video   = OUTPUT_DIR / f'{output_path}.mp4'
  positions_path = OUTPUT_DIR / f'{output_path}-positions.json'

  with open(CORNERS_PATH) as f:
    corners = np.array(json.load(f), dtype=np.float32)
  H = np.load(HOMOGRAPHY_PATH)

  cap = cv2.VideoCapture(str(input_path))
  assert cap.isOpened(), f"Could not open {input_path}"

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
  writer = cv2.VideoWriter(str(output_video), fourcc, fps, (out_w, out_h))

  all_positions = {}
  frame_idx = 0

  print(f"Processing {frame_count} frames at {fps:.0f} fps...")
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    balls = detect_fn(frame, table_mask)
    translated = transformBalls(balls, H)

    if translated:
      all_positions[frame_idx] = [(tx, ty) for tx, ty in translated]

    out_frame = drawFrame(frame, corners, balls, translated)
    writer.write(out_frame)

    if frame_idx % 500 == 0:
      print(f"  Frame {frame_idx}/{frame_count} — {len(balls)} balls detected")

    frame_idx += 1

  cap.release()
  writer.release()

  with open(positions_path, 'w') as f:
    json.dump(all_positions, f)

  print(f"Output video: {output_video}")
  print(f"Positions log: {positions_path} ({len(all_positions)} frames with detections)")


if __name__ == '__main__':
  from homography.detection import detectBalls
  #processVideo(detectBalls, VIDEO_PATH_1, 'recording-felt-output')
  #processVideo(detectBalls, VIDEO_PATH_1, 'recording-felt-output')
  processVideo(detectBalls, VIDEO_PATH, 'recording-felt-output')
