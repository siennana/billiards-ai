import numpy as np
import cv2 as cv
from pathlib import Path

_HERE = Path(__file__).parent

# read image
img = cv.imread(str(_HERE / 'images' / 'pooltable-raw-scrnsht.PNG'))
assert img is not None, "file could not be read, check with os.path.exists()"
# display image
cv.imshow('image', img)
k = cv.waitKey(0)
# write image
if k == ord('s'):  # 's' key
  cv.imwrite(str(_HERE / 'images' / 'pooltable-saved.PNG'), img)
# access pixel value by row and column coordinates
px = img[100, 100]
print(px)
# access image properties
print(img.shape)  # height, width, number of channels
print(img.size)   # total number of pixels
print(img.dtype)  # data type of the image

