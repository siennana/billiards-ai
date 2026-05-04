import cv2
import numpy as np
import json
from pathlib import Path

from homography import transformBalls
from stat_tracking import PocketTracker, standardPockets

_HERE = Path(__file__).parent

VIDEO_PATH      = _HERE.parent / 'video' / 'recording.mkv'
VIDEO_PATH_1      = _HERE.parent / 'video' / 'recording-1.mkv'
CORNERS_PATH    = _HERE.parent / 'data' / 'homography' / 'corners.json'
HOMOGRAPHY_PATH = _HERE.parent / 'data' / 'homography' / 'homography.npy'
OUTPUT_DIR      = _HERE.parent / 'video' / 'test-output'

OUTPUT_WIDTH  = 450
OUTPUT_HEIGHT = 900

# 16 visually distinct BGR colors for tracking up to 16 ball IDs.
# Indexed by `track_id % 16` so wraparound is graceful when the tracker
# assigns a fresh ID after losing/regaining a ball.
BALL_COLORS = [
  (0, 255, 255),   # yellow
  (255, 0, 0),     # blue
  (0, 0, 255),     # red
  (128, 0, 128),   # purple
  (0, 165, 255),   # orange
  (0, 200, 0),     # green
  (40, 80, 140),   # maroon
  (200, 200, 200), # light gray (stand-in for 8-ball)
  (255, 255, 0),   # cyan
  (255, 0, 255),   # magenta
  (0, 255, 128),   # spring green
  (180, 180, 255), # pink
  (255, 200, 0),   # azure
  (50, 150, 50),   # forest green
  (200, 100, 220), # lavender
  (240, 240, 240), # off-white (stand-in for cue)
]


# Draws the original frame with table outline and detected ball markers,
# and a top-down view with translated ball positions.
#
# When tracePaths=True, balls is expected to be 4-tuples (cx, cy, r, id) and
# translated is 3-tuples (tx, ty, id). trails_orig and trails_top map
# track_id -> list of past (x, y) points and are drawn as colored polylines.
def drawFrame(frame, corners, balls, translated, tracePaths=False,
              trails_orig=None, trails_top=None):
  # --- Left panel: original with overlays ---
  left = frame.copy()
  pts = corners.astype(np.int32)
  cv2.polylines(left, [pts], isClosed=True, color=(255, 255, 255), thickness=1)

  if tracePaths and trails_orig:
    for bid, points in trails_orig.items():
      if len(points) >= 2:
        cv2.polylines(left, [np.array(points, dtype=np.int32)],
                      isClosed=False, color=BALL_COLORS[bid % 16], thickness=2)

  for ball in balls:
    cx, cy, r = ball[0], ball[1], ball[2]
    color = BALL_COLORS[ball[3] % 16] if tracePaths and len(ball) > 3 else (0, 255, 255)
    cv2.circle(left, (int(cx), int(cy)), int(r), color=color, thickness=2)

  # --- Right panel: top-down view ---
  right = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
  cv2.rectangle(right, (0, 0), (OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1),
                color=(255, 255, 255), thickness=2)

  if tracePaths and trails_top:
    for bid, points in trails_top.items():
      if len(points) >= 2:
        cv2.polylines(right, [np.array(points, dtype=np.int32)],
                      isClosed=False, color=BALL_COLORS[bid % 16], thickness=2)

  for tball in translated:
    tx, ty = tball[0], tball[1]
    color = BALL_COLORS[tball[2] % 16] if tracePaths and len(tball) > 2 else (0, 255, 255)
    cv2.circle(right, (tx, ty), 6, color=color, thickness=-1)

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
# detect_fn signature:
#   tracePaths=False: (frame, table_mask) -> list[(cx, cy, r)]
#   tracePaths=True:  (frame, table_mask) -> list[(cx, cy, r, ball_id)]
# When tracePaths=True the per-id position history is accumulated and drawn
# as colored polylines on both panels.
def processVideo(detect_fn, input_path, output_path, tracePaths=False, trackStats=False):
  output_video   = OUTPUT_DIR / f'{output_path}.mp4'
  positions_path = OUTPUT_DIR / f'{output_path}-positions.json'
  events_path    = OUTPUT_DIR / f'{output_path}-events.json'

  # Pocket detection requires ball IDs, which only the tracking path provides.
  if trackStats and not tracePaths:
    raise ValueError("trackStats=True requires tracePaths=True (needs ball IDs)")
  pocket_tracker = (PocketTracker(standardPockets(OUTPUT_WIDTH, OUTPUT_HEIGHT))
                    if trackStats else None)

  with open(CORNERS_PATH) as f:
    data = json.load(f)
  corners = np.array(data["corners"] if isinstance(data, dict) else data, dtype=np.float32)
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
  trails_orig = {}
  trails_top  = {}
  frame_idx = 0

  print(f"Processing {frame_count} frames at {fps:.0f} fps...")
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    detections = detect_fn(frame, table_mask)

    if tracePaths:
      # Strip ids before homography (transformBalls expects 3-tuples), then
      # re-pair by index — order is preserved.
      xy_only = [(cx, cy, r) for cx, cy, r, _ in detections]
      translated_xy = transformBalls(xy_only, H)
      translated = [(tx, ty, bid)
                    for (tx, ty), (_, _, _, bid) in zip(translated_xy, detections)]
      balls = detections

      for cx, cy, _, bid in balls:
        trails_orig.setdefault(bid, []).append((int(cx), int(cy)))
      for tx, ty, bid in translated:
        trails_top.setdefault(bid, []).append((int(tx), int(ty)))

      if translated:
        all_positions[frame_idx] = [(tx, ty, bid) for tx, ty, bid in translated]

      if pocket_tracker is not None:
        for ev in pocket_tracker.update(frame_idx, translated):
          print(f"  POCKET: frame {ev['frame']} ball #{ev['ball_id']} -> pocket {ev['pocket_index']}")
    else:
      balls = detections
      translated = transformBalls(balls, H)
      if translated:
        all_positions[frame_idx] = [(tx, ty) for tx, ty in translated]

    out_frame = drawFrame(frame, corners, balls, translated, tracePaths,
                          trails_orig if tracePaths else None,
                          trails_top  if tracePaths else None)
    writer.write(out_frame)

    if frame_idx % 100 == 0:
      print(f"  Frame {frame_idx}/{frame_count} — {len(balls)} balls detected")

    frame_idx += 1

  cap.release()
  writer.release()

  with open(positions_path, 'w') as f:
    json.dump(all_positions, f)

  print(f"Output video: {output_video}")
  print(f"Positions log: {positions_path} ({len(all_positions)} frames with detections)")

  if pocket_tracker is not None:
    with open(events_path, 'w') as f:
      json.dump(pocket_tracker.events, f, indent=2)
    print(f"Pocket events: {events_path} ({len(pocket_tracker.events)} events)")


if __name__ == '__main__':
  from detection.balls import detectBallsHSV
  processVideo(detectBallsHSV, VIDEO_PATH, 'recording-felt-output')
