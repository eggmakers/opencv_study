import cv2
import numpy as np

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, ((int(img.shape[1]/2), int(img.shape[0]/2))), interpolation=cv2.INTER_AREA)

img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height,width = img_GRAY.shape[0:2]

img_GrayUP = np.zeros((height, width), np.uint8)
img_GrayDown = np.zeros((height, width), np.uint8)

#图像灰度增强
for i in range(height):
    for j in range(width):
        if (int(img_GRAY[i, j] + 70) > 255):
            gray_up = 255
        else:
            gray_up = int(img_GRAY[i, j] + 70)
        img_GrayUP[i, j] = np.uint8(gray_up)
        
#图像灰度减弱
for i in range(height):
    for j in range(width):
        if (int(img_GRAY[i, j] - 70) < 0):
            gray_down = 0
        else:
            gray_down = int(img_GRAY[i, j] - 70)
        img_GrayDown[i, j] = np.uint8(gray_down)
        
cv2.imshow("Origin Image", img_GRAY)
cv2.imshow("UP Gray", img_GrayUP)
cv2.imshow("Down Gray", img_GrayDown)
cv2.waitKey()
cv2.destroyAllWindows()