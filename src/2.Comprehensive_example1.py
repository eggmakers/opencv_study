#在图像上添加一个图像
import cv2

img1 = cv2.imread("picture_material/batman.jpg")
img1 = cv2.resize(img1, (int(img1.shape[1]/2),int(img1.shape[0]/2)), interpolation=cv2.INTER_AREA)
img2 = cv2.imread("picture_material/logo.jpg")
img2 = cv2.resize(img2, (int(img2.shape[1]/4),int(img2.shape[0]/4)), interpolation=cv2.INTER_AREA)

#创建ROI
rows1,cols1,channels1 = img1.shape
rows,cols,channels = img2.shape
roi = img1[0:rows, (cols1-cols):cols1]

#创建logo掩码，并同时创建相反掩码
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(img2gray, 240, 255, cv2.THRESH_BINARY)
#mask背景是白色，彩色logo是黑色
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi, roi, mask = mask)
img2_bg = cv2.bitwise_and(img2,img2, mask = mask_inv)

#将logo放入ROI，并修正主图像
dst = cv2.add(img1_bg,img2_bg)
img1[0:rows, (cols1 - cols):cols1] = dst

cv2.imshow("Result",img1)
cv2.waitKey()