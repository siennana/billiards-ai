### Ideas
* Draw Mode
    * allow user to draw patterns on table and translate into "runnable pattern with description"

### Goal
* gather data from video stream
    * track balls made/missed
    * identy types: bank, kick, safety, etc.
    * suggestions: bank/kick angles
* translate video stream to a simple animation representing the game state

### Plan
* PHASE 1:
    * given video stream, record the following:
        * balls made, balls missed, total time stamp
    * store in database
* PHASE 2:


### Process
* OpenCV homography and basic object detection
    * calibrate table dimensions from static image
    * use homography to translate raw dimensions into 2D rectangular plane

* train YOLO model
    * create custom dataset with Roboflow