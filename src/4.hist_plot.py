import cv2
from matplotlib import pyplot as plt

img = cv2.imread("picture_material/beauty_leg3.jpg")
img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
color = ('b', 'r', 'g')
plt.figure(1)
plt.subplot(121), plt.imshow(img_RGB) 
plt.subplot(122),plt.hist(img.ravel(), 256, [0, 255])
plt.figure(2)
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
plt.show()