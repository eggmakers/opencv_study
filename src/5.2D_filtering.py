import cv2
import numpy as np

img = cv2.imread("picture_material/beauty_leg3.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#自定义的卷积函数
kernel3 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
kernel5 = np.array([[-1, -1, -1, -1, 0], [-1, -1, -1, 0, 1], [-1, -1, 0, 1, 1], [-1, 0, 1, 1, 1], [0, 1, 1, 1, 1]])

img3 = cv2.filter2D(img, -1, kernel3)
img5 = cv2.filter2D(img, -1, kernel5)

cv2.imshow("Origin", img)
cv2.imshow("K3", img3)
cv2.imshow("K5", img5)

cv2.waitKey()
