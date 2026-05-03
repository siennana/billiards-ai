import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# COCO class index for "sports ball" — what the pretrained YOLOv8 model
# generally fires on for billiard balls.
COCO_SPORTS_BALL = 32
YOLO_WEIGHTS = Path(__file__).parent.parent.parent / 'yolo' / 'weights' / 'yolov8n.pt'
YOLO_CONF    = 0.25
# Reject YOLO detections whose derived radius exceeds this — guards against
# the occasional huge bbox the model fires on pocket shadows or felt artifacts.
MAX_BALL_RADIUS = 30

_yolo_model = None


def _getYoloModel():
  global _yolo_model
  if _yolo_model is None:
    from ultralytics import YOLO
    _yolo_model = YOLO(str(YOLO_WEIGHTS))
  return _yolo_model


# HSV range for the blue table felt — used by the felt-subtraction detector
FELT_LOWER = np.array([85, 20, 80])
FELT_UPPER = np.array([135, 180, 220])

# Contour filters for the HSV felt-based detector
HSV_MIN_AREA = 80
HSV_MAX_AREA = 800
HSV_MIN_CIRCULARITY = 0.5


# Detects billiard balls by masking the blue felt and finding ball-sized,
# roughly-circular blobs in what remains. No deep model — purely color-based.
# Returns (cx, cy, radius) tuples matching the detector interface.
def detectBallsHSV(frame, table_mask):
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
    if area < HSV_MIN_AREA or area > HSV_MAX_AREA:
      continue
    perim = cv2.arcLength(c, True)
    if perim == 0:
      continue
    circularity = 4 * np.pi * area / (perim ** 2)
    if circularity < HSV_MIN_CIRCULARITY:
      continue
    (cx, cy), radius = cv2.minEnclosingCircle(c)
    balls.append((float(cx), float(cy), float(radius)))

  return balls


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


# Detects billiard balls using a YOLOv8 model pretrained on COCO, filtering
# to the "sports ball" class. Bounding boxes are converted to (cx, cy, r)
# where r = max(w, h) / 2. Keeps only detections whose centers fall inside
# table_mask. Returns a list of (cx, cy, radius) tuples.
def detectBallsYOLO(frame, table_mask):
  model = _getYoloModel()
  results = model(frame, classes=[COCO_SPORTS_BALL], conf=YOLO_CONF, verbose=False)

  if not results or results[0].boxes is None or len(results[0].boxes) == 0:
    return []

  h, w = table_mask.shape[:2]
  balls = []
  for box in results[0].boxes.xywh.cpu().numpy():
    cx, cy, bw, bh = box
    r = max(bw, bh) / 2.0
    if r > MAX_BALL_RADIUS:
      continue
    ix, iy = int(cx), int(cy)
    if 0 <= iy < h and 0 <= ix < w and table_mask[iy, ix] > 0:
      balls.append((float(cx), float(cy), float(r)))

  return balls


# Tracks billiard balls across frames using a trained YOLOv8 model with
# ultralytics' built-in tracker (BoT-SORT by default). Must be called
# frame-by-frame in temporal order so `persist=True` keeps track IDs stable.
# Accepts either a file path string or a pre-loaded YOLO instance — pass a
# loaded model to avoid reloading weights on every frame. Filters to class 0
# ('ball'). Returns (cx, cy, r, ball_id) tuples.
def trackBallsYoloTrained(frame, table_mask, model):
  if isinstance(model, str):
    model = YOLO(model)

  results = model.track(frame, persist=True, verbose=False)

  if not results or results[0].boxes is None or results[0].boxes.id is None:
    return []

  h, w = table_mask.shape[:2]
  boxes = results[0].boxes
  ids = boxes.id.cpu().numpy().astype(int)
  classes = boxes.cls.cpu().numpy().astype(int)
  xywh = boxes.xywh.cpu().numpy()

  balls = []
  for (cx, cy, bw, bh), bid, cls in zip(xywh, ids, classes):
    if cls != 0:
      continue
    r = max(bw, bh) / 2.0
    if r > MAX_BALL_RADIUS:
      continue
    ix, iy = int(cx), int(cy)
    if 0 <= iy < h and 0 <= ix < w and table_mask[iy, ix] > 0:
      balls.append((float(cx), float(cy), float(r), int(bid)))

  return balls


# Detects billiard balls using a trained YOLOv8 model. Accepts either a file
# path string or a pre-loaded YOLO instance (pass a loaded model to avoid
# reloading weights on every frame). Filters to class 0 ('ball') and centers
# that fall inside table_mask. Returns (cx, cy, radius) tuples.
def detectBallsYoloTrained(frame, table_mask, model):
  if isinstance(model, str):
    model = YOLO(model)

  results = model(frame, verbose=False)[0]
  h, w = table_mask.shape[:2]
  balls = []
  for box in results.boxes:
    if int(box.cls) != 0:
      continue
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    r = ((x2 - x1) + (y2 - y1)) / 4
    if r > MAX_BALL_RADIUS:
      continue
    ix, iy = int(cx), int(cy)
    if 0 <= iy < h and 0 <= ix < w and table_mask[iy, ix] > 0:
      balls.append((cx, cy, r))

  return balls
