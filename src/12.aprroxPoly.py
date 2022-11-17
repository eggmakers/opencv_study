import cv2
import numpy as np

img = cv2.imread("picture_material/shazam.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 0 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, 3, 2)
cnt = contours[0]

#多边形逼近，得到角点
epsilon1 = 0.1 * cv2.arcLength(cnt, True)
epsilon2 = 0.01 * cv2.arcLength(cnt, True)
epsilon3 = 0.001 * cv2.arcLength(cnt, True)

approx1 = cv2.approxPolyDP(cnt, epsilon1, True)
approx2 = cv2.approxPolyDP(cnt, epsilon2, True)
approx3 = cv2.approxPolyDP(cnt, epsilon3, True)

#画出多边形
image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# cv2.imshow("origin", img)
image1 = cv2.drawContours(image, [approx1], 0, (0, 0, 255), 2)
cv2.imshow("approxPloyDP 10%", image1)
image2 = cv2.drawContours(image, [approx2], 0, (255, 0, 0), 2)
cv2.imshow("approxPloyDP 1%", image2)
image3 = cv2.drawContours(image, [approx3], 0, (0, 255, 255), 2)
cv2.imshow("approxPloyDP 0.1%", image3)
cv2.waitKey()