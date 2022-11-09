import cv2
import numpy as np
from skimage import io, color
import skimage.morphology as sm
from matplotlib import pyplot as plt

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

plt.figure('MORPH_OPEN_CLOSE', figsize=(8, 8))
plt.subplot(131)
plt.title('Origin')
plt.imshow(img)
plt.axis('off')

plt.subplot(132)
plt.title('Opening')
plt.imshow(opening)
plt.axis('off')

plt.subplot(133)
plt.title('closing')
plt.imshow(closing)
plt.axis('off')
plt.show()

#skimage函数处理
img = color.rgb2gray(img)
dst = sm.opening(img, sm.disk(5))#直径为4的圆形滤波器进行开运算
dst1 = sm.closing(img, sm.disk(5))#直径为5的圆形滤波器进行开运算

plt.figure('Morphology_Open_Close', figsize=(8, 8))
plt.subplot(131)
plt.title('Origin')
plt.imshow(img)
plt.axis('off')

plt.subplot(132)
plt.title('Opening Image')
plt.imshow(dst)
plt.axis('off')

plt.subplot(133)
plt.title('Closing Image')
plt.imshow(dst1)
plt.axis('off')
plt.show()