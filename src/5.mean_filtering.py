import cv2
from skimage import util

img = cv2.imread("picture_material/beauty_leg3.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
img = util.random_noise(img, mode = 's&p')              #椒盐噪声
#当以不同大小的卷积核
img1 = cv2.blur(img, (3, 3))        #卷积核为3*3，实现均值滤波
img2 = cv2.blur(img, (7, 7))
img3 = cv2.blur(img, (15, 15))
cv2.imshow('Origin', img)
#显示滤波后的图像
cv2.imshow('N = 3', img1)
cv2.imshow('N = 7', img2)
cv2.imshow('N = 15', img3)
cv2.waitKey()