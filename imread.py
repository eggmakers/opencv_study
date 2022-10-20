import cv2
import os
img = cv2.imread("face_2.jfif")
print(img)
cv2.imshow("img",img)
cv2.waitKey()
cv2.destroyAllWindows()