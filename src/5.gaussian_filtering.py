import cv2
from skimage import util

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
img = util.random_noise(img, mode = 's&p')              #椒盐噪声

img3 = cv2.GaussianBlur(img, (3, 3), 0, 0)      #卷积核3 x 3
img7 = cv2.GaussianBlur(img, (7, 7), 0, 0)      #卷积核7 x 7
img15 = cv2.GaussianBlur(img, (15, 15), 0, 0)    #卷积核15 x 15

cv2.imshow("Origin",img)
cv2.imshow("N = 3 Gauss", img3)
cv2.imshow("N = 7 Gauss", img7)
cv2.imshow("N = 15 Gauss", img15)
cv2.waitKey()
