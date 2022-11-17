import cv2
import numpy as np

img = cv2.imread("picture_material/shazam.jpg")
# img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

draw_img = img.copy()

#画出轮廓
ret = cv2.drawContours(draw_img, contours, 0, (0, 0, 255), 2)
cnt = contours[0]
cv2.imshow("draw", ret)
#画出直角矩形
x, y, w, h = cv2.boundingRect(cnt)
img_rect = cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
#画出旋转矩形
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
min_rect = cv2.drawContours(ret, [box], 0, (0, 255, 0), 2)

cv2.imshow("Rectangles", min_rect)

k = cv2.waitKey()
if k == 27:
    cv2.destroyAllWindows()