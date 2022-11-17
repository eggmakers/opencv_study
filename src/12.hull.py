import cv2

img = cv2.imread("picture_material/shazam.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, 2, 1)
cnt = contours[0]

hull = cv2.convexHull(cnt, True, False)
#绘制外接线
length = len(hull)
for i in range(len(hull)):
    cv2.line(img, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 255, 0), 2)
#显示坐标点
cv2.drawContours(img, hull, -1, (0, 0, 255), 3)

cv2.imshow("line & points", img)
cv2.waitKey()