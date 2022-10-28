import cv2
import numpy as np
from matplotlib import pyplot as plt

#---镜像变换---#
img1 = cv2.imread("picture_material/beauty_leg3.jpg")
img1 = cv2.resize(img1, (int(img1.shape[1]/2), int(img1.shape[0]/2)), interpolation=cv2.INTER_AREA)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)#翻转时图像会变成BGR图像
img2 = cv2.flip(img1, 1)
img3 = cv2.flip(img1, 0)
img4 = cv2.flip(img1, -1)
#---镜像变换end---#

#---透视变换---#
img5 = cv2.imread("picture_material/3.png")
img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)
print(img5.shape)
#选择合适的数据点
pts1 = np.float32([[500, 200], [4000, 200], [100, 1500], [4550, 1400]])
pts2 = np.float32([[0, 0], [4750, 0], [0, 1500], [4750, 1500]])
#创建M透视变换矩阵
M = cv2.getPerspectiveTransform(pts1, pts2)
#透视变换
dst = cv2.warpPerspective(img5, M, (4750, 1500))
#---透视变换end---#

#---镜像变换---#
plt.subplot(221), plt.imshow(img1)
plt.axis('off'), plt.title("Original")
plt.subplot(222), plt.imshow(img2)
plt.axis('off'), plt.title("Horizontal")
plt.subplot(223), plt.imshow(img3)
plt.axis('off'), plt.title("Vertical")
plt.subplot(224), plt.imshow(img4)
plt.axis('off'), plt.title("Diagonal")
# plt.show()
#---镜像变换end---#

#---透视变换---#
plt.subplot(121),plt.imshow(img5),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
#---透视变换end---#