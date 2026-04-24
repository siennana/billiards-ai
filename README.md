# billiards-ai

## PHASE 1
This phase demonstrates how a homography maps the camera's perspective view to a bird's-eye view that is easier for game analysis.

### Image pipeline

1. Raw input photo: an original camera image of the table.
2. Pre-homography overlay: the same image with the user-calibrated table corners, table boundary, and ball centers marked.
3. Post-homography output: the normalized top-down game state where detected locations are translated into a consistent coordinate frame.

### Visual demonstration

Raw image:

![Raw table image](images/table-snapshot-raw.jpg)

Pre-homography overlay:

![Pre-homography overlay](images/pre-homography-overlay.jpg)

Post-homography output:

![Post-homography top-down output](images/post-homography-detection-output.jpg)

### Video demonstration

![Video demonstration](public/recording-output-clip.gif)

