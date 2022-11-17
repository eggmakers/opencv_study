import cv2
import numpy as np

img = cv2.imread("picture_material/beauty_leg3.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
b, g, r = cv2.split(img)

cv2.imshow("Origin", img)
cv2.imshow("Blue", b)
cv2.imshow("Green", g)
cv2.imshow("Red", r)
#通道扩展
zeros = np.zeros(img.shape[:2], np.uint8)
img_B = cv2.merge([b, zeros, zeros])
img_G = cv2.merge([zeros, g, zeros])
img_R = cv2.merge([zeros, zeros, r])
cv2.imshow("B_img", img_B)
cv2.imshow("G_img", img_G)
cv2.imshow("R_img", img_R)
cv2.waitKey()