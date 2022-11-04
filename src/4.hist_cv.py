import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread("picture_material/beauty_leg3.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
#mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[300:750, 40:400] = 255
masked_img = cv2.bitwise_and(img, img, mask = mask)

hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(221), plt.axis('off'), plt.imshow(img, 'gray')
plt.subplot(222), plt.axis('off'), plt.imshow(masked_img, 'gray')
plt.subplot(223), plt.plot(hist_full)
plt.subplot(224), plt.plot(hist_mask)
plt.xlim([0, 256])
plt.show()