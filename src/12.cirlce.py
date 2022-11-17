import cv2
import numpy as np

img = cv2.imread("picture_material/shazam.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

draw_img = img.copy()
ret = cv2.drawContours(draw_img, contours, 0, (0, 0, 255), 2)
#取出单个轮廓值
cnt = contours[0]

(x, y), radius = cv2.minEnclosingCircle(cnt)
centers = (int(x), int(y))
radius = int(radius)
cv2.circle(ret, centers, radius, (0, 255, 255), 2)
ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(ret, ellipse, (0, 255, 0), 2)

cv2.imshow("draw circle", ret)
cv2.waitKey()