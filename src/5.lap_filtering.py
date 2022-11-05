import cv2
img = cv2.imread("picture_material/beauty_leg3.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)

lap = cv2.Laplacian(img, cv2.CV_16S, ksize = 3)

#将图像转换为8bit图像
laplacian = cv2.convertScaleAbs(lap)

cv2.imshow("Origin",img)
cv2.imshow("Laplacian",laplacian)

cv2.waitKey()