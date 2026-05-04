import json
import sys
from pathlib import Path


# Standard 6-pocket positions in top-down (rectangular) table coordinates:
# 4 corners + 2 side pockets at the midpoint of each long side.
def standardPockets(width, height):
  return [
    (0,     0),           # 0: top-left corner
    (width, 0),           # 1: top-right corner
    (0,     height),      # 2: bottom-left corner
    (width, height),      # 3: bottom-right corner
    (0,     height // 2), # 4: middle-left side pocket
    (width, height // 2), # 5: middle-right side pocket
  ]


# Detects ball-pocket events from a stream of per-frame ball positions.
# Pure logic — doesn't know about video, detection, or rendering.
#
# Feed it (frame_idx, balls) each frame, where balls is a list of
# (tx, ty, ball_id) in top-down table coordinates. It fires an event when a
# tracked ball stops appearing AND its last known position was within
# pocket_radius of any pocket AND it has been gone for at least `patience`
# frames (to absorb temporary tracker dropouts).
#
# Tunables:
#   pocket_radius     — how close to a pocket counts as "in" (top-down pixels)
#   patience          — frames the ball must be missing before firing
#   min_track_length  — drop tracks shorter than this; they're usually tracker
#                       flicker, not real balls (avoids false pocket events)
class PocketTracker:
  def __init__(self, pockets, pocket_radius=30, patience=3, min_track_length=5):
    self.pockets = pockets
    self.pocket_radius_sq = pocket_radius * pocket_radius
    self.patience = patience
    self.min_track_length = min_track_length

    self.last_seen  = {}    # ball_id -> (frame_idx, tx, ty)
    self.first_seen = {}    # ball_id -> frame_idx
    self.resolved   = set() # ball_ids we've already decided on (pocketed or noise)
    self.events     = []    # accumulated pocket events

  # Returns the list of NEW pocket events fired on this frame (usually empty).
  def update(self, frame_idx, balls):
    current_ids = set()
    for tx, ty, bid in balls:
      current_ids.add(bid)
      self.first_seen.setdefault(bid, frame_idx)
      self.last_seen[bid] = (frame_idx, tx, ty)

    new_events = []
    for bid, (last_frame, tx, ty) in list(self.last_seen.items()):
      if bid in current_ids or bid in self.resolved:
        continue
      if frame_idx - last_frame < self.patience:
        continue

      # Track too short to trust — likely tracker noise, not a real ball.
      if last_frame - self.first_seen[bid] < self.min_track_length:
        self.resolved.add(bid)
        continue

      pocket_idx = self._nearest_pocket(tx, ty)
      if pocket_idx is None:
        # Disappeared mid-table (tracker drop, occlusion). Mark resolved so
        # we don't keep checking, but emit nothing.
        self.resolved.add(bid)
        continue

      event = {
        "frame":         last_frame,
        "ball_id":       int(bid),
        "pocket_index":  pocket_idx,
        "last_position": [int(tx), int(ty)],
      }
      new_events.append(event)
      self.events.append(event)
      self.resolved.add(bid)

    return new_events

  def _nearest_pocket(self, x, y):
    best_idx = None
    best_d = self.pocket_radius_sq
    for i, (px, py) in enumerate(self.pockets):
      d = (x - px) ** 2 + (y - py) ** 2
      if d <= best_d:
        best_d = d
        best_idx = i
    return best_idx


# Offline mode: run pocket detection on a positions JSON produced by
# video_process.processVideo (with tracePaths=True). Useful for retuning
# patience/radius without re-processing the video.
#
# Usage: python stat_tracking.py <positions.json> [width] [height]
if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Usage: stat_tracking.py <positions.json> [width=450] [height=900]")
    sys.exit(1)

  positions_path = Path(sys.argv[1])
  width  = int(sys.argv[2]) if len(sys.argv) > 2 else 450
  height = int(sys.argv[3]) if len(sys.argv) > 3 else 900

  with open(positions_path) as f:
    positions = json.load(f)

  tracker = PocketTracker(standardPockets(width, height))

  # JSON keys are frame indices as strings; values are lists of [tx, ty, bid]
  for frame_idx_str in sorted(positions.keys(), key=int):
    frame_idx = int(frame_idx_str)
    balls = [(tx, ty, bid) for tx, ty, bid in positions[frame_idx_str]]
    tracker.update(frame_idx, balls)

  events_path = positions_path.with_name(positions_path.stem.replace('-positions', '') + '-events.json')
  with open(events_path, 'w') as f:
    json.dump(tracker.events, f, indent=2)

  print(f"Pocket events: {len(tracker.events)}")
  for e in tracker.events:
    print(f"  frame {e['frame']}: ball #{e['ball_id']} -> pocket {e['pocket_index']} at {e['last_position']}")
  print(f"Saved: {events_path}")
