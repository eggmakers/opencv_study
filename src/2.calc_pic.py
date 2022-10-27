import cv2
import numpy as np
img1 = cv2.imread("picture_material/beauty_leg2.jpg")
img2 = cv2.imread("picture_material/beauty_leg3.jpg")

img1 = cv2.resize(img1, (int(img1.shape[1]/2),int(img1.shape[0]/2)), interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, (int(img1.shape[1]),int(img1.shape[0])), interpolation=cv2.INTER_AREA)

data = 2 * np.ones([int(img1.shape[0]), int(img1.shape[1]), 3], np.uint8)

img3 = cv2.add(img1, img2)
img4 = img1 + img2
img5 = cv2.addWeighted(img1,0.7,img2,0.5,0)
img6 = cv2.subtract(img1, img2)
img7 = img1 - img2

img8 = cv2.multiply(img1,data)
img9 = cv2.divide(img1,data)

cv2.imshow('Image1',img1)
cv2.imshow('Image2',img2)
cv2.imshow('result1 = cv2.add(img1,img2)',img3)
cv2.imshow('result2 = img1 + img2',img4)
cv2.imshow('result3 = cv2.addWeighted(img1+img2)',img5)
cv2.imshow('result4 = img1 - img2',img7)
cv2.imshow('result5 = cv2.subtract(img1+img2)',img6)
cv2.imshow('result6 = cv2.multiply(img1)',img8)
cv2.imshow('result7 = cv2.divide(img1)',img9)

cv2.waitKey(0)