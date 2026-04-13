import cv2 as cv
from pathlib import Path

_HERE = Path(__file__).parent

# Load the image
image = cv.imread(str(_HERE.parent / 'images' / 'pooltable-raw-scrnsht.PNG'))

# Convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv.Canny(gray, 100,50)

# Overlay edges in red on the original image
overlay = image.copy()
overlay[edges != 0] = (0, 0, 255)

# Save the output image
out_path = _HERE / 'images' / 'canny-edges.PNG'
cv.imwrite(str(out_path), overlay)

print(f"Canny edge detection completed. Output saved as '{out_path}'")