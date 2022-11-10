import cv2

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
#创建矩阵结构元
se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#腐蚀图像
r = cv2.erode(img, se, iterations = 1)
e = img - r

cv2.imshow('origin', img)
cv2.imshow('erode_img', r)
#显示边界提取的效果
cv2.imshow('edge', e)
cv2.waitKey()