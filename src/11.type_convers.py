import cv2
img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
cv2.imshow("Origin", img)
cv2.imshow("HSV", img_HSV)
cv2.imshow("YCrCb", img_YCrCb)
cv2.waitKey()