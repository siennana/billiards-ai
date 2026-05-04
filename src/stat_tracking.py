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


# Detects ball-pocket events AND shot events from a stream of per-frame ball
# positions. Pure logic — doesn't know about video, detection, or rendering.
#
# Feed it (frame_idx, balls) each frame, where balls is a list of
# (tx, ty, ball_id) in top-down table coordinates.
#
# Pocket events: a tracked ball stops appearing AND its last known position
# was within pocket_radius of any pocket AND it has been gone for at least
# `patience` frames (to absorb temporary tracker dropouts).
#
# Shot events: a window starting when any ball moves more than
# movement_threshold pixels and ending after settle_frames consecutive frames
# of stillness. Each shot records which balls moved, ball counts at start and
# end, and a type derived from the count delta.
#
# Tunables:
#   pocket_radius      — how close to a pocket counts as "in" (top-down px)
#   patience           — frames the ball must be missing before firing
#   min_track_length   — drop tracks shorter than this; they're usually tracker
#                        flicker, not real balls (avoids false pocket events)
#   movement_threshold — per-frame pixel delta that counts as "moving"
#   settle_frames      — consecutive still frames required to close a shot
class PocketTracker:
  def __init__(self, pockets, pocket_radius=30, patience=3, min_track_length=5,
               fps=None, movement_threshold=3, settle_frames=25):
    self.pockets = pockets
    self.pocket_radius_sq = pocket_radius * pocket_radius
    self.patience = patience
    self.min_track_length = min_track_length
    self.fps = fps
    self.movement_threshold_sq = movement_threshold * movement_threshold
    self.settle_frames = settle_frames

    self.last_seen  = {}    # ball_id -> (frame_idx, tx, ty)
    self.first_seen = {}    # ball_id -> frame_idx
    self.resolved   = set() # ball_ids we've already decided on (pocketed or noise)
    self.events     = []    # accumulated pocket events
    self.shotEvents = []    # accumulated shot events

    # Shot-detection state
    self._prev_positions          = {}     # bid -> (tx, ty) from last frame
    self._in_shot                 = False
    self._shot_start_frame        = None
    self._shot_begin_total        = None
    self._shot_moving_ids         = set()
    self._shot_last_motion_frame  = None
    self._frames_since_movement   = 0

  # Returns the list of NEW pocket events fired on this frame (usually empty).
  def update(self, frame_idx, balls):
    current_ids = set()
    current_positions = {}
    for tx, ty, bid in balls:
      current_ids.add(bid)
      current_positions[bid] = (tx, ty)
      self.first_seen.setdefault(bid, frame_idx)
      self.last_seen[bid] = (frame_idx, tx, ty)

    self._updateShot(frame_idx, current_ids, current_positions)

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
        "totalBalls":    len(current_ids),
      }
      if self.fps:
        event["timestamp"] = self._formatTimestamp(last_frame)
      new_events.append(event)
      self.events.append(event)
      self.resolved.add(bid)

    return new_events

  # Call once after the final frame to flush any shot still in progress.
  def finalize(self):
    if self._in_shot:
      self._closeShot(end_total=len(self._prev_positions))

  def _updateShot(self, frame_idx, current_ids, current_positions):
    moving_now = set()
    for bid, (tx, ty) in current_positions.items():
      prev = self._prev_positions.get(bid)
      if prev is None:
        continue
      ptx, pty = prev
      if (tx - ptx) ** 2 + (ty - pty) ** 2 > self.movement_threshold_sq:
        moving_now.add(bid)

    if moving_now:
      if not self._in_shot:
        self._in_shot          = True
        self._shot_start_frame = frame_idx
        self._shot_begin_total = len(current_ids)
        self._shot_moving_ids  = set()
      self._shot_moving_ids.update(moving_now)
      self._shot_last_motion_frame = frame_idx
      self._frames_since_movement  = 0
    elif self._in_shot:
      self._frames_since_movement += 1
      if self._frames_since_movement >= self.settle_frames:
        self._closeShot(end_total=len(current_ids))

    self._prev_positions = current_positions

  def _closeShot(self, end_total):
    begin_total = self._shot_begin_total
    if end_total < begin_total:
      shot_type = "ballLost"
    elif end_total > begin_total:
      shot_type = "ballAdded"
    else:
      shot_type = "ballSame"

    end_frame       = self._shot_last_motion_frame
    duration_frames = end_frame - self._shot_start_frame

    shot = {
      "type":            shot_type,
      "startFrame":      self._shot_start_frame,
      "endFrame":        end_frame,
      "movedBallIds":    sorted(int(b) for b in self._shot_moving_ids),
      "beginTotalBalls": begin_total,
      "endTotalBalls":   end_total,
    }
    if self.fps:
      shot["timestamp"] = self._formatTimestamp(self._shot_start_frame)
      shot["duration"]  = round(duration_frames / self.fps, 2)
    self.shotEvents.append(shot)

    self._in_shot                = False
    self._shot_start_frame       = None
    self._shot_begin_total       = None
    self._shot_moving_ids        = set()
    self._shot_last_motion_frame = None
    self._frames_since_movement  = 0

  def _formatTimestamp(self, frame):
    total_s = int(frame / self.fps)
    h, rem  = divmod(total_s, 3600)
    m, s    = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

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
# Usage: python stat_tracking.py <positions.json> [width] [height] [fps]
if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Usage: stat_tracking.py <positions.json> [width=450] [height=900] [fps]")
    sys.exit(1)

  positions_path = Path(sys.argv[1])
  width  = int(sys.argv[2]) if len(sys.argv) > 2 else 450
  height = int(sys.argv[3]) if len(sys.argv) > 3 else 900
  fps    = float(sys.argv[4]) if len(sys.argv) > 4 else None

  with open(positions_path) as f:
    positions = json.load(f)

  tracker = PocketTracker(standardPockets(width, height), fps=fps)

  # JSON keys are frame indices as strings; values are lists of [tx, ty, bid]
  for frame_idx_str in sorted(positions.keys(), key=int):
    frame_idx = int(frame_idx_str)
    balls = [(tx, ty, bid) for tx, ty, bid in positions[frame_idx_str]]
    tracker.update(frame_idx, balls)
  tracker.finalize()

  events_path = positions_path.with_name(positions_path.stem.replace('-positions', '') + '-events.json')
  with open(events_path, 'w') as f:
    json.dump({
      "pocketEvents": tracker.events,
      "shotEvents":   tracker.shotEvents,
    }, f, indent=2)

  print(f"Pocket events: {len(tracker.events)}")
  for e in tracker.events:
    print(f"  frame {e['frame']}: ball #{e['ball_id']} -> pocket {e['pocket_index']} at {e['last_position']}")
  print(f"Shot events: {len(tracker.shotEvents)}")
  for s in tracker.shotEvents:
    print(f"  frame {s['startFrame']}-{s['endFrame']}: {s['type']} "
          f"({s['beginTotalBalls']}->{s['endTotalBalls']}, moved {s['movedBallIds']})")
  print(f"Saved: {events_path}")
