import cv2

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 50, 135,cv2.THRESH_BINARY)
#获得轮廓
contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#画出轮廓
draw_img = img.copy()
ret = cv2.drawContours(draw_img, contours, -1, (0, 255, 255), 2)
cv2.imshow('Contour img', ret)
cv2.waitKey()