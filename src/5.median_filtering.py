import cv2
from skimage import util

img = cv2.imread("result/origin5.png")

img3 = cv2.medianBlur(img, 3)
img7 = cv2.medianBlur(img, 7)
img15 = cv2.medianBlur(img, 15)

cv2.imshow("Origin",img)
cv2.imshow("N = 3 median", img3)
cv2.imshow("N = 7 median", img7)
cv2.imshow("N = 15 median", img15)
cv2.waitKey()
