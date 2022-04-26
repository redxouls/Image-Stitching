import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_left = cv.imread("warp1.jpg")
img_right = cv.imread("warp.jpg")

plt.imshow(img_left, alpha=0.5)
plt.imshow(img_right, alpha=0.5)
plt.show()