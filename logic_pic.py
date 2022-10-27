import cv2
import numpy as np

img1 = cv2.imread("picture_material/beauty_leg2.jpg")
img1_5 = cv2.imread("picture_material/beauty_leg3.jpg")
img1 = cv2.resize(img1, (int(img1.shape[1]/2),int(img1.shape[0]/2)), interpolation=cv2.INTER_AREA)
img1_5 = cv2.resize(img1_5, (int(img1_5.shape[1]/2),int(img1_5.shape[0]/2)), interpolation=cv2.INTER_AREA)
img2 = np.zeros(img1.shape,dtype = np.uint8)
img2[465:755, 147:441] = 255
img3 = cv2.bitwise_and(img1, img2)
img4 = cv2.bitwise_or(img1, img2)
img5 = cv2.bitwise_not(img1_5)
img6 = cv2.bitwise_xor(img1, img2)

cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("img3",img3)
cv2.imshow("img4",img4)
cv2.imshow("img5",img5)
cv2.imshow("img6",img6)

cv2.waitKey()