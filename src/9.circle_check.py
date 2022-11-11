import cv2
import numpy as np

img = cv2.imread("picture_material/coin.jpg")
# img = cv2.resize(img, (int(img.shape[1]*5), int(img.shape[0]*5)), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("origin", gray)

#hough变换
circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 
100, param1 = 100, param2 = 30, minRadius = 180, maxRadius = 185)
circles = circles1[0, :, :]
circles = np.uint16(np.around(circles))
for i in circles[:]:
    cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 5)#画圆
    cv2.circle(img, (i[0], i[1]), 2, (255, 0, 255), 10)#画圆心
    
cv2.imshow("result",img)
cv2.waitKey()