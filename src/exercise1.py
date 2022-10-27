#编写程序，对图片进行读取，显示和保存
import cv2
img1 = cv2.imread("picture_material/beauty_leg2.jpg")
img1 = cv2.resize(img1, (int(img1.shape[1]/2), int(img1.shape[0]/2)), interpolation = cv2.INTER_AREA)
cv2.imshow("show",img1)
cv2.imwrite("result/save.jpg",img1)

cv2.waitKey()

#编写程序，实现对图像的算术运算
img2  = cv2.imread("picture_material/beauty_leg3.jpg")
img2 = cv2.resize(img2, (int(img1.shape[1]), int(img1.shape[0])), interpolation = cv2.INTER_AREA)

##加减法运算
img_add = cv2.add(img1, img2)
cv2.imshow("img_add",img_add)
img_sub = cv2.subtract(img1,img2)
cv2.imshow("img_sub",img_sub)

##乘除法运算
img_multi = cv2.multiply(img1, img2)
cv2.imshow("img_multi",img_multi)
img_div = cv2.divide(img1,img2)
cv2.imshow("img_div",img_div)

cv2.waitKey()

#编写程序，实现对图像的逻辑运算
img_and = cv2.bitwise_and(img1, img2)
cv2.imshow("img_and",img_and)
img_or = cv2.bitwise_or(img1, img2)
cv2.imshow("img_or",img_or)
img_not = cv2.bitwise_not(img1)
cv2.imshow("img_not",img_not)
img_xor= cv2.bitwise_xor(img1, img2)
cv2.imshow("img_xor",img_xor)

cv2.waitKey()

#编写程序，用电脑摄像头捕获一张图片
import numpy as np
img3 = cv2.VideoCapture(0)
while True:
    sucess,img4 = img3.read()
    cv2.imshow("img3",img4)

    k = cv2.waitKey(1)
    
    if k == 27:#按下esc
        cv2.destroyAllWindows()
        break
    elif k == ord("s"):
        cv2.imwrite("result/capture.jpg",img4)
        cv2.destroyAllWindows()
        break
img3.release()

#编写程序，将图标放置在图像的左上角
img5 = cv2.imread("picture_material/batman.jpg")
img6 = cv2.imread("picture_material/logo.jpg")

img5 = cv2.resize(img5, (int(img5.shape[1]/2),int(img5.shape[0]/2)), interpolation = cv2.INTER_AREA)
img6 = cv2.resize(img6, (int(img6.shape[1]/4), int(img6.shape[0]/4)), interpolation = cv2.INTER_AREA)

rows1,cols1,channels1 = img5.shape
rows,cols,channels  = img6.shape
roi = img5[0:rows, 0:(cols - cols1)]

img2gray = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(img2gray, 240, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi, roi, mask = mask)
img2_bg = cv2.bitwise_and(img6, img6, mask = mask_inv)

dst = cv2.add(img1_bg,img2_bg)
img5[0:rows, 0:(cols - cols1)] = dst

cv2.imshow("Result",img5)
cv2.waitKey()
