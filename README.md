# billiards-ai

## PHASE 1
This phase demonstrates how a homography maps the camera's perspective view to a bird's-eye view that is easier for game analysis.

### Image pipeline

1. Raw input photo: an original camera image of the table.
2. Pre-homography overlay: the same image with the user-calibrated table corners, table boundary, and ball centers marked.
3. Post-homography output: the normalized top-down game state where detected locations are translated into a consistent coordinate frame.

### Ball Detection and Tracking with YOLOv8 Trained Model

![Detection demonstration](public/ben-sienna-20260503_232606-cfr-clip2_yolov5-detected.gif)

![Tracked demonstration](public/ben-sienna-20260503_232606-cfr-clip2_yolo-v5-tracked.gif)

