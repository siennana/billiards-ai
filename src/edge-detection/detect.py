import cv2 as cv
import numpy as np
from pathlib import Path

_HERE = Path(__file__).parent
IMAGE_PATH = _HERE.parent.parent / 'images' / 'pooltable-painted-scrnsht.PNG'
OUTPUT_PATH = _HERE / 'images' / 'pooltable-detected.PNG'

# Paint colors used to mark the table
# #be1251 (pink)  → BGR(81, 18, 190) → HSV(169, 231, 190)
# #1fb356 (green) → BGR(86, 179, 31) → HSV(71,  211, 179)
RAIL_HSV_LOWER  = np.array([155, 150,  80])
RAIL_HSV_UPPER  = np.array([179, 255, 220])


def _fit_segment(contour):
    """Fit a straight line to a contour and return its two endpoints."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    [vx, vy, x0, y0] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])
    # Project every point onto the line direction to find the extent
    t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
    pt1 = (int(x0 + t.min() * vx), int(y0 + t.min() * vy))
    pt2 = (int(x0 + t.max() * vx), int(y0 + t.max() * vy))
    return pt1, pt2


def detectRails(img):
    """
    Detect the 6 pool table rails from an image painted with #be1251 (pink) rail markers.
    Returns a list of 6 (pt1, pt2) tuples, each a straight line segment for one rail.
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    pink_mask = cv.inRange(hsv, RAIL_HSV_LOWER, RAIL_HSV_UPPER)

    # Close small gaps in the painted strokes
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    pink_mask = cv.morphologyEx(pink_mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(pink_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Keep the 6 largest contours (one per rail), discard paint specks
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:6]

    return [_fit_segment(c) for c in contours]


def detectPockets(_img, _rails):
    """
    Detect the 6 pocket openings from an image painted with #1fb356 (green) pocket markers.
    Returns a list of (cx, cy) pocket center points.
    """
    raise NotImplementedError("detectPockets not yet implemented")


if __name__ == '__main__':
    img = cv.imread(str(IMAGE_PATH))
    assert img is not None, f"Could not read {IMAGE_PATH}"

    rails = detectRails(img)
    print(f"Rails found: {len(rails)}")
    for i, (p1, p2) in enumerate(rails):
        print(f"  Rail {i+1}: {p1} -> {p2}")

    canvas = img.copy()
    for p1, p2 in rails:
        cv.line(canvas, p1, p2, color=(0, 220, 0), thickness=2)

    cv.imwrite(str(OUTPUT_PATH), canvas)
    print(f"Saved: {OUTPUT_PATH}")
