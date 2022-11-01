#编写程序，使用cv2.warpAffine()函数将一幅图像放大为原来的1/2后，顺时针旋转90°，再向左平移20个单位，向上平移20单位，不改变图像大小
from this import d
import cv2
import numpy as np

img1 = cv2.imread("picture_material/beauty_leg3.jpg")

h,w = img1.shape[0:2]
M = np.float32([[1, 0, -20], [0, 1, -20]])

dst = cv2.resize(img1, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

cols,rows = dst.shape[0:2]
M1 = cv2.getRotationMatrix2D((cols/2, rows/2), 90, -1)

dst1 = cv2.warpAffine(dst, M1, (cols, rows), borderValue=(255,255,255))

dst2 = cv2.warpAffine(dst1, M, (2 * cols, 2 * rows))

cv2.imshow("resize",dst)
cv2.imshow("rotate",dst1)
cv2.imshow("pan",dst2)

cols = int(cols / 2)
rows = int(rows / 2)


#编写程序，将一幅图像剪切为源图像的一半
img2 = dst[0:rows, 0:cols]
cv2.imshow("img cut",img2)


#编写程序，使用cv2。flip()函数对一幅图像分别进行x轴镜像变换，y轴镜像变换以及对角变换
dst3 = cv2.flip(dst, 0)
dst4 = cv2.flip(dst, 1)
dst5 = cv2.flip(dst, -1)
cv2.imshow("x axis transformation", dst3)
cv2.imshow("y axis transformation", dst4)
cv2.imshow("Diagonal transformation", dst5)

cv2.waitKey()