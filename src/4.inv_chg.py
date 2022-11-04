import cv2
import numpy as np

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, ((int(img.shape[1]/2), int(img.shape[0]/2))), interpolation=cv2.INTER_AREA)

img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height,width = img_Gray.shape[0: 2]

color_change = np.zeros((height,width), np.uint8)

for i in range(height):
    for j in range(width):
        gray = 255 - img_Gray[i, j]
        color_change[i, j] = np.uint8(gray)
        
img_out = 255 - img
        
cv2.imshow("Gray", img_Gray)
cv2.imshow("RGB",img)
cv2.imshow("Color_change", color_change)
cv2.imshow("Color_change_RGB", img_out)
cv2.waitKey()
cv2.destroyAllWindows()