from ast import Is
from tkinter import N
import cv2
import numpy as np
from matplotlib import pyplot as plt
    
img1 = cv2.imread("picture_material/beauty_leg2.jpg")
img1 = cv2.resize(img1, (int(img1.shape[1]/2), int(img1.shape[0]/2)), interpolation=cv2.INTER_AREA)
img2 = cv2.imread("picture_material/1.jpg")#仿射
img3 = img1[465:755, 147:441]

M = np.float32([[1, 0, 50], [0, 1, 25]])
rows,cols = img1.shape[0:2]#平移
rows1,cols1,channels1 = img2.shape#仿射
rows2,cols2 = img1.shape[0:2]#旋转
h, w, c = img1.shape#旋转

#旋转45°
M1 = cv2.getRotationMatrix2D((cols2/2, rows2/2), 45, 1)
M2 = cv2.getRotationMatrix2D((cols2/2, rows2/2), 45, -1)
# center:旋转中心
# angle:旋转角度
# scale:缩放比例以及旋转方向，正数为逆时针，负数为顺时针
M3 = np.zeros((2, 3), dtype = np.float32)
alpha = np.cos(np.pi / 4.0)
beta = np.sin(np.pi / 4.0)

#将rows和cols转置
dst = cv2.warpAffine(img1, M, (2 * cols, 2 * rows))
dst1 = cv2.warpAffine(img1, M, (cols, rows))

#设置缩放比例
dst2 = cv2.resize(img1, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
#设置图像大小
width,height = img1.shape[0:2]
dst3 = cv2.resize(img1, (height,width), interpolation = cv2.INTER_LANCZOS4)

#旋转
dst4 = cv2.warpAffine(img1, M1, (cols, rows), borderValue=(255,255,255))
dst5 = cv2.warpAffine(img1, M2, (cols, rows), borderValue=(255,255,255))

#初始化旋转矩阵
M[0, 0] = alpha
M[1, 1] = alpha
M[0, 1] = beta
M[1, 0] = -beta
cx = w / 2
cy = h / 2
tx = (1 - alpha) * cx - beta * cy
ty = beta * cx + (1 - alpha) * cy
M[0, 2] = tx
M[1, 2] = ty

#更改为全尺寸
bound_w = int(h * np.abs(beta) + w * np.abs(alpha))
bound_h = int(h * np.abs(alpha) + w * np.abs(beta))

#添加中心位置迁移
M[0, 2] = bound_w / 2 - cx
M[1, 2] = bound_h / 2 - cy
dst6 = cv2.warpAffine(img1, M3, (bound_w, bound_h))

#选择合适的数据点（仿射）
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

#创建仿射变换矩阵（仿射）
M = cv2.getAffineTransform(pts1,pts2)

cv2.imshow("origin",img1)
cv2.imshow("image panning",dst)#显示窗口变大的平移
cv2.imshow("image panning2",dst1)#显示窗口不变的平移
cv2.imshow("Image scaling1",dst2)#缩小图像
cv2.imshow("Image scaling2",dst3)#图像大小不变
cv2.imshow("Image rotation1",dst4)#图像逆时针旋转45°
cv2.imshow("Image rotation2",dst5)#图像顺时针旋转45°
cv2.imshow("Image rotation3",dst6)#图像无剪切旋转
cv2.imshow("Image cut",img3)#图像剪切

#仿射变换
dst = cv2.warpAffine(img2, M, (cols1, rows1))

plt.subplot(121), plt.imshow(img2), plt.title('Affine transformation Input')
plt.subplot(122), plt.imshow(dst), plt.title('Affine transformation Output')
plt.show()

cv2.waitKey()
