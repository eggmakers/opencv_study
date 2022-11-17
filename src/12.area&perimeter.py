import cv2

img = cv2.imread("picture_material/coin.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
countours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

draw_img = img.copy()
ret = cv2.drawContours(draw_img, countours, 0, (0, 0, 255), 2)

cnt = countours[0]#取单个轮廓值
area = cv2.contourArea(cnt)
length = cv2.arcLength(cnt, True)
print('Area = ', area, "Length = ", length)
cv2.imshow('ret', ret)
cv2.waitKey()
