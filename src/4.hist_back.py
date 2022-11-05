import cv2
import numpy as np
roi = cv2.imread("result/img3.jpg")
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

target = cv2.imread("picture_material/beauty_leg2.jpg")
target = cv2.resize(target,(int(target.shape[1]/2), int(target.shape[0]/2)), interpolation=cv2.INTER_AREA)
hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

#计算对象的直方图
roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

#直方图归一化并利用反传算法
cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)

#卷积滤波
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cv2.filter2D(dst, -1, disc, dst)

#应用阈值做与操作
ret, thresh = cv2.threshold(dst, 50, 255, 0)
thresh = cv2.merge((thresh, thresh, thresh))
back_projection = cv2.bitwise_and(target, thresh)

cv2.imshow('Origin', roi)
cv2.imshow('Target', target)
cv2.imshow('Thresh', thresh)
cv2.imshow('Backprojection', back_projection)

cv2.waitKey()