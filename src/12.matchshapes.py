import cv2
import numpy as np

img = cv2.imread("picture_material/number.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#二值化
ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#搜索轮廓
contours, hierarchy = cv2.findContours(thresh, 3, 2)
hierarchy = np.squeeze(hierarchy)

#载入模板
img_8 = cv2.imread("picture_material/8.jpg")
img_8 = cv2.cvtColor(img_8, cv2.COLOR_BGR2GRAY)
ret1, th = cv2.threshold(img_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours1, hierarchy1 = cv2.findContours(th, 3, 2)
template_8 = contours1[0]

#记录最匹配的值的大小和位置
min_pos = 0
min_value = 9
for i in range(len(contours)):
    value = cv2.matchShapes(template_8, contours[i], 1, 0.0)
    if value < min_value:
        min_value = value
        min_pos = i
        
cv2.drawContours(gray, [contours1[min_pos]], 0, [0, 0, 255], 2)
cv2.imshow("result", gray)
cv2.waitKey()