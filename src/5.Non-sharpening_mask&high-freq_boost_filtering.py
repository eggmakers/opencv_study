import cv2
import numpy as np

def fun_uh(img, k):
    imgBlur = cv2.GaussianBlur(img, (3, 3), 0, 0)
    imgMask = img - imgBlur
    res = img + np.uint8(k * imgMask)
    return res

img = cv2.imread("picture_material/1.jpg")

#非锐化掩模，k=1
mask_img = fun_uh(img, 1)
#高频提升滤波，k=3
high_img = fun_uh(img, 3)

cv2.imshow("Origin",img)
cv2.imshow("Unsharp mask",mask_img)
cv2.imshow("High freq",high_img)
cv2.waitKey()