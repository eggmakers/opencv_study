import cv2
import numpy as np
import pytesseract as pt

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#创建haar级联器
plater = cv2.CascadeClassifier('haarcascade files\haarcascade_russian_plate_number.xml')

#导入图片，将其灰度化
img = cv2.imread("picture_material/pai.jpg")
img = cv2.GaussianBlur(img, (3, 3), 5)
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#进行人脸识别
#数据类型[x, y, w, h]
plate = plater.detectMultiScale(gray, 1.1, 5)
for (x, y, w, h) in plate:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)

roi_img = gray[y:y + h, x:x + w]
ret, roi_bin = cv2.threshold(roi_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(pt.image_to_string(roi_bin, lang = 'chi_sim+eng', config = '--psm 8 --oem 3'))

cv2.imshow("roi_bin", roi_bin)
cv2.imshow("plate", img)
cv2.waitKey()