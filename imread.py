import cv2
import os
img = cv2.imread("picture_material/beauty_leg2.jpg")
cv2.namedWindow("img")
cv2.imshow("img",img)
print(img.shape[1])
resized = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
cv2.imshow('Resized Image',resized)
print(resized.shape)
cv2.waitKey()