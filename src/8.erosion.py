import cv2
import numpy as np
from skimage import data
import skimage.morphology as sm
from matplotlib import pyplot as plt

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations = 1)

plt.figure('erode', figsize=(8, 8))
plt.subplot(121)
plt.title('Origin')
plt.imshow(img)
plt.axis('off')

plt.subplot(122)
plt.title('Erode Image')
plt.imshow(erosion)
plt.axis('off')
plt.show()

#用skimage实现腐蚀
img = data.checkerboard()
dst1 = sm.erosion(img, sm.square(5))    #用边长为5的正方形进行腐蚀滤波
dst2 = sm.erosion(img, sm.square(15))   #用边长为15的正方形进行腐蚀滤波

plt.figure('morphology', figsize = (8, 8))
plt.subplot(131)
plt.title('Origin')
plt.imshow(img, plt.cm.gray)

plt.subplot(132)
plt.title('5*5 eroded image')
plt.imshow(dst1, plt.cm.gray)

plt.subplot(133)
plt.title('15*15 eroded image')
plt.imshow(dst2, plt.cm.gray)
plt.show()