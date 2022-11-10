import cv2

img = cv2.imread("picture_material/beauty_leg2.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilate_img = cv2.dilate(img, kernel)
erode_img = cv2.erode(img, kernel)

absdiff_img = cv2.absdiff(dilate_img, erode_img)#获得边缘
#二值化观察结果
retval, threshold_img = cv2.threshold(absdiff_img, 40, 255, cv2.THRESH_BINARY)
#反色
result = cv2.bitwise_not(threshold_img)

cv2.imshow('Origin', img)
cv2.imshow('dilate', dilate_img)
cv2.imshow('erode', erode_img)
cv2.imshow('absdiff', absdiff_img)
cv2.imshow('threshold', threshold_img)
cv2.imshow('result', result)
cv2.waitKey()
