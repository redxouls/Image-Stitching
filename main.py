from PIL import Image
from PIL.ExifTags import TAGS

import cv2
import numpy as np
import matplotlib.pyplot as plt

base_name = "DSC029"
start = 59
end = 76

img = cv2.imread("./images/DSC02976.JPG", cv2.IMREAD_COLOR)


scale_percent = 10 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)
print(len(keypoints_1))
print(keypoints_1)
cv2.drawKeypoints(gray, keypoints_1, img)

cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

