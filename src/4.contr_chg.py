import cv2
import numpy as np

img = cv2.imread("picture_material/beauty_leg3.jpg")
img = cv2.resize(img, ((int(img.shape[1]/2), int(img.shape[0]/2))), interpolation=cv2.INTER_AREA)

img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = img_Gray.shape[0: 2]

Contrast_enhancement = np.zeros((height, width), np.uint8)#创建新图像
Contrast_reduction = np.zeros((height, width), np.uint8)

#对比度增强，k = 1.3
for i in range(height):
    for j in range(width):
        if (int(img_Gray[i, j] * 1.3) > 255):
            gray = 255
        else:
            gray = int(img_Gray[i, j] * 1.3)
        Contrast_enhancement[i, j] = np.uint8(gray)

#对比度减弱, k = 0.5
for i in range(height):
    for j in range(width):
        if (int(img_Gray[i, j] * 0.5) < 0):
            gray = 0
        else:
            gray = int(img_Gray[i, j] * 0.5)
        Contrast_reduction[i, j] = np.uint8(gray)
        
cv2.imshow("Gray", img_Gray)
cv2.imshow("Enhancement", Contrast_enhancement)
cv2.imshow("Reduction", Contrast_reduction)
cv2.waitKey()
cv2.destroyAllWindows()