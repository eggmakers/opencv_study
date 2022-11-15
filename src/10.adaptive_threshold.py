import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret,th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

titles = ['Origin', 'Binary', 'Mean', 'Gauss']
images = [img, th1, th2, th3]
plt.figure('adaptiveThreshold')
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.show()