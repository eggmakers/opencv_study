import cv2
import numpy as np

#创建haar级联器
faser = cv2.CascadeClassifier('haarcascade files\haarcascade_frontalface_default.xml')
eyer = cv2.CascadeClassifier('haarcascade files\haarcascade_eye.xml')

#导入图片，将其灰度化
img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#进行人脸识别
#数据类型[x, y, w, h]
faces = faser.detectMultiScale(gray, 1.1, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)
    
#进行眼睛识别
eyes = eyer.detectMultiScale(gray, 1.1, 5)
for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
cv2.imshow("face", img)
cv2.waitKey()