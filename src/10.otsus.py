import cv2
from matplotlib import pyplot as plt
import numpy as np

def sp_noise(img, prob):
    #椒盐噪声 prob：噪声比例
    img_noise = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rNum = np.random.random()
            if rNum < prob:
                img_noise[i][j] = 0
            elif rNum > thres:
                img_noise[i][j] = 255
            else:
                img_noise[i][j] = img[i][j]
    return img_noise

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = sp_noise(img, 0.06)
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret2, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#先高斯滤波，再otsus二值化
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

images = [img, 0, th1, img, 0, th2, blur, 0, th3]
titles = ['origin', 'Histogram', 'Threshold', 'origin', 'Histogram', 'Otsus', 'GaussianBlur', 'Histogram', 'Otsus']
for i in range(3):
    plt.subplot(3, 3, i * 3 + 1),plt.imshow(images[i * 3]),plt.title(titles[i * 3]),plt.axis('off')
    plt.subplot(3, 3, i * 3 + 2),plt.hist(images[i * 3].ravel(), 256),plt.title(titles[i * 3 + 1])
    plt.subplot(3, 3, i * 3 + 3),plt.imshow(images[i * 3 + 2]),plt.title(titles[i * 3 + 2]),plt.axis('off')
plt.show()