import cv2

src = cv2.imread("picture_material/1.jpg")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)

#十字结构
se = cv2.getStructuringElement(cv2.MORPH_CROSS, (11, 11), (-1, -1))
#击中与击不中
binary = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, se)

cv2.imshow('origin', src)
cv2.imshow('hit_miss', binary)
cv2.waitKey()