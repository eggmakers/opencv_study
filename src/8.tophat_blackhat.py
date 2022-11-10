import cv2
import numpy as np
from skimage import io, color
import skimage.morphology as sm
from matplotlib import pyplot as plt

img = cv2.imread("picture_material/1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.ones((9, 9), np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

plt.figure('MORPH_TOPHAT', figsize=(8, 8))
plt.subplot(131)
plt.title('Origin')
plt.imshow(img)
plt.axis('off')

plt.subplot(132)
plt.title('tophat image')
plt.imshow(tophat)
plt.axis('off')

plt.subplot(133)
plt.title('blackhat image')
plt.imshow(blackhat)
plt.axis('off')
plt.show()

#skimage函数运算
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst1 = sm.white_tophat(img, sm.square(30))
dst2 = sm.black_tophat(img, sm.square(30))
plt.figure('morphology', figsize=(8, 8))
plt.subplot(131)
plt.title('Origin')
plt.imshow(img)
plt.axis('off')

plt.subplot(132)
plt.title('tophat image')
plt.imshow(dst1)
plt.axis('off')

plt.subplot(133)
plt.title('blackhat image')
plt.imshow(dst2)
plt.axis('off')
plt.show()