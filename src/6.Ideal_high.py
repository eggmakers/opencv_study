import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("picture_material/beauty_leg3.jpg")
img = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_float = np.float32(img)
#傅里叶变换
dft = cv2.dft(img_float, flags = cv2.DFT_COMPLEX_OUTPUT)
#将低频转移到中心
dft_center = np.fft.fftshift(dft)
#定义掩模，中间为0，周围为1
crow, ccol = int(img.shape[0] / 2),int(img.shape[1] / 2)
#设置掩模区域10*10
mask = np.zeros((img.shape[0], img.shape[1], 2), np.uint8)
mask[crow - 5 : crow + 5, ccol - 5 : ccol + 5] = 0
#设置掩模区域100*100
mask1 = np.zeros((img.shape[0], img.shape[1], 2), np.uint8)
mask1[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 0
#图像相乘，保留中间部分
mask_img = dft_center * mask
mask1_img = dft_center * mask1
#将低频移动到原来位置
img_idf = np.fft.ifftshift(mask_img)
img1_idf = np.fft.ifftshift(mask1_img)
#傅里叶逆变换
img_idf = cv2.idft(img_idf)
img1_idf = cv2.idft(img1_idf)
#将其转换成空间域
img_idf = cv2.magnitude(img_idf[:, :, 0], img_idf[:, :, 1])
img1_idf = cv2.magnitude(img1_idf[:, :, 0], img1_idf[:, :, 1])
#输出图像
plt.figure("Ideal high pass")
plt.subplot(131),plt.title("Origin")
plt.imshow(img, cmap = 'gray'),plt.axis('off')
plt.subplot(132),plt.title("Highpass mask = 5")
plt.imshow(img_idf, cmap = 'gray'),plt.axis('off')
plt.subplot(133),plt.title("Highpass mask = 30")
plt.imshow(img1_idf, cmap = 'gray'),plt.axis('off')
plt.show()