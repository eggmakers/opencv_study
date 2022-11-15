import cv2

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
cv2.imshow("origin", img)

ret, img1 = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", img1)

#二值图像反色
height, width = img1.shape[0: 2]
img2 = img1.copy()
for i in range(height):
    for j in range(width):
        img2[i, j] = (255 - img1[i, j])
cv2.imshow("inv", img2)

img2_3 = cv2.Canny(img2, 80, 255)
cv2.imshow("canny", img2_3)

height1, width1 = img2_3.shape
img3 = img2_3.copy()
for i in range(height1):
    for j in range(width1):
        img3[i, j] = (255 - img2_3[i, j])
        
cv2.imshow("inv_canny", img3)
cv2.waitKey()    