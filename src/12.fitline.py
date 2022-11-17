import cv2
import numpy as np

img = cv2.imread("picture_material/shazam.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, 3, 2)
cnt = contours[0]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
print([vx, vy, x, y])

rows, cols = img.shape[:2]
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
img = cv2.line(img, (cols - 1, righty), (0, lefty), (0, 0, 255), 2)

cv2.imshow("Line", img)
cv2.waitKey()
