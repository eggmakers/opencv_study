#编写程序，使用不同的卷积核对带有高斯噪声的图像进行均值滤波，方框滤波和高斯滤波，观察其效果
import cv2
from skimage import util
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = np.array(img)

noise_gs_img = util.random_noise(img, mode = 'gaussian')                #高斯噪声
img_mean3 = cv2.blur(noise_gs_img, (3, 3))  
img_mean7 = cv2.blur(noise_gs_img, (7, 7))
img_mean15 = cv2.blur(noise_gs_img, (15, 15))

img_box1 = cv2.boxFilter(noise_gs_img, -1, (3, 3), normalize = 1)        #无归一化处理
img_box2 = cv2.boxFilter(noise_gs_img, -1, (2, 2), normalize = 0)        #有归一化处理

img_gauss3 = cv2.GaussianBlur(img, (3, 3), 0, 0)                         #卷积核3 x 3
img_gauss7 = cv2.GaussianBlur(img, (7, 7), 0, 0)                         #卷积核7 x 7
img_gauss15 = cv2.GaussianBlur(img, (15, 15), 0, 0)                      #卷积核15 x 15

plt.figure("filtering1")
plt.subplot(331), plt.imshow(img)
plt.axis('off'), plt.title("gauss_noise")
plt.subplot(332), plt.imshow(img_box1)
plt.axis('off'), plt.title("box_none")
plt.subplot(333), plt.imshow(img_box2)
plt.axis('off'), plt.title("box")
plt.subplot(334), plt.imshow(img_mean3)
plt.axis('off'), plt.title("mean 3*3")
plt.subplot(335), plt.imshow(img_mean7)
plt.axis('off'), plt.title("mean 7*7")
plt.subplot(336), plt.imshow(img_mean15)
plt.axis('off'), plt.title("mean 15*15")
plt.subplot(337), plt.imshow(img_gauss3)
plt.axis('off'), plt.title("gauss 3*3")
plt.subplot(338), plt.imshow(img_gauss7)
plt.axis('off'), plt.title("gauss 7*7")
plt.subplot(339), plt.imshow(img_gauss15)
plt.axis('off'), plt.title("gauss 15*15")
plt.show()

#编写程序，使用不同的卷积核对带有椒盐噪声，高斯噪声的图像进行中值滤波和双边滤波，观察其效果
img = cv2.imread("picture_material/beauty_leg3.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.array(img)

noise_gs_img = util.random_noise(img, mode = 'gaussian')                #高斯噪声
noise_sp_img = util.random_noise(img, mode = 's&p')                     #椒盐噪声

img_median_gs3 = cv2.medianBlur(noise_gs_img, 3)
img_median_gs7 = cv2.medianBlur(noise_gs_img, 7)
img_median_gs15 = cv2.medianBlur(noise_gs_img, 15)

img_median_sp3 = cv2.medianBlur(noise_sp_img, 3)
img_median_sp7 = cv2.medianBlur(noise_sp_img, 7)
img_median_sp15 = cv2.medianBlur(noise_sp_img, 15)

img_bilater_gs30 = cv2.bilateralFilter(noise_gs_img, 30, 50, 100)            #滤波半径30
img_bilater_gs70 = cv2.bilateralFilter(noise_gs_img, 70, 50, 100)            #滤波半径70
img_bilater_gs150 = cv2.bilateralFilter(noise_gs_img, 150, 50, 100)          #滤波半径150

img_bilater_sp30 = cv2.bilateralFilter(noise_sp_img, 30, 50, 100)            #滤波半径30
img_bilater_sp70 = cv2.bilateralFilter(noise_sp_img, 70, 50, 100)            #滤波半径70
img_bilater_sp150 = cv2.bilateralFilter(noise_sp_img, 150, 50, 100)          #滤波半径150

plt.figure("filtering2")
plt.subplot(5, 3, 1), plt.imshow(img)
plt.axis('off'), plt.title("origin")
plt.subplot(5, 3, 2), plt.imshow(noise_gs_img)
plt.axis('off'), plt.title("gauss_noise")
plt.subplot(5, 3, 3), plt.imshow(noise_sp_img)
plt.axis('off'), plt.title("sp_noise")
plt.subplot(5, 3, 4), plt.imshow(img_median_gs3)
plt.axis('off'), plt.title("median_filtering_gs 3*3")
plt.subplot(5, 3, 5), plt.imshow(img_median_gs7)
plt.axis('off'), plt.title("median_filtering_gs 7*7")
plt.subplot(5, 3, 6), plt.imshow(img_median_gs15)
plt.axis('off'), plt.title("median_filtering_gs 15*15")
plt.subplot(5, 3, 7), plt.imshow(img_median_sp3)
plt.axis('off'), plt.title("median_filtering_sp 3*3")
plt.subplot(5, 3, 8), plt.imshow(img_median_sp7)
plt.axis('off'), plt.title("median_filtering_sp 7*7")
plt.subplot(5, 3, 9), plt.imshow(img_median_sp15)
plt.axis('off'), plt.title("median_filtering_sp 15*15")
plt.subplot(5, 3, 10), plt.imshow(img_bilater_gs30)
plt.axis('off'), plt.title("median_bilater_gs 30")
plt.subplot(5, 3, 11), plt.imshow(img_bilater_gs70)
plt.axis('off'), plt.title("median_bilater_gs 70")
plt.subplot(5, 3, 12), plt.imshow(img_bilater_gs150)
plt.axis('off'), plt.title("median_bilater_gs 150")
plt.subplot(5, 3, 13), plt.imshow(img_bilater_sp30)
plt.axis('off'), plt.title("median_bilater_sp 30")
plt.subplot(5, 3, 14), plt.imshow(img_bilater_sp70)
plt.axis('off'), plt.title("median_bilater_sp 70")
plt.subplot(5, 3, 15), plt.imshow(img_bilater_sp150)
plt.axis('off'), plt.title("median_bilater_sp 150")

plt.show()