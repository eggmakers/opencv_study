import cv2
import os
img = cv2.imread("picture_material/beauty_leg2.jpg")
cv2.namedWindow("img")
cv2.imshow("img",img)
cv2.destroyAllWindows()
resized = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
cv2.imshow('Resized Image',resized)
print(resized.shape)
print(resized.size)
print(resized.dtype)
cv2.imwrite("picture_material/resized.jpg",resized)
cv2.waitKey()
