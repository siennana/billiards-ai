import cv2 as cv

# Load the image
image = cv.imread('pooltable-raw-scrnsht.PNG')

# Convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv.Canny(gray, 100,50)

# Overlay edges in red on the original image
overlay = image.copy()
overlay[edges != 0] = (0, 0, 255)

# Save the output image
cv.imwrite('canny-edges.PNG', overlay)

print("Canny edge detection completed. Output saved as 'canny-edges.PNG'")