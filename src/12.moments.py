import cv2
import numpy as np

img = cv2.imread("picture_material/beauty_leg2.jpg")
cv2.imshow("origin", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

n = len(contours)
contoursImg = []
for i in range(n):
    temp = np.zeros(img.shape, np.uint8)
    contoursImg.append(temp)
    contoursImg[i] = cv2.drawContours(contoursImg[i], contours, i, (0, 255, 255), 3)
    # cv2.imshow("contours[" + str(i) + "]", contoursImg[i])
    
print("moments are:")
for i in range(n):
    print("contours" + str(i) + "\n", cv2.moments(contours[i]))
print("areas are:")
for i in range(n):
    print("area is " + str(i) + "%d" %cv2.moments(contours[i])['m00'])
    
cv2.waitKey()