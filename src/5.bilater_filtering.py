import cv2
from skimage import util

img = cv2.imread("result/origin5.png")

img1 = cv2.bilateralFilter(img, 30, 50, 100)            #滤波半径30
img2 = cv2.bilateralFilter(img, 70, 50, 100)            #滤波半径70
img3 = cv2.bilateralFilter(img, 150, 50, 100)           #滤波半径150

cv2.imshow("Origin",img)
cv2.imshow("BF1", img1)
cv2.imshow("BF2", img2)
cv2.imshow("BF3", img3)
cv2.waitKey()
