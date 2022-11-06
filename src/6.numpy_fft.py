import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#傅里叶变换
fft_img = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft_img)
fft_res = np.log(np.abs(fft_shift))

#傅里叶逆变换
ifft_shift = np.fft.fft2(fft_shift)
ifft_img = np.fft.fftshift(ifft_shift)
ifft_img = np.log(np.abs(ifft_img))

plt.figure("傅里叶变换")
plt.subplot(131),plt.imshow(img, 'gray')
plt.title("Origin"),plt.axis('off')
plt.subplot(132),plt.imshow(fft_res, 'gray')
plt.title("Fourier"),plt.axis('off')
plt.subplot(133),plt.imshow(ifft_img, 'gray')
plt.title("Inverse Fourier"),plt.axis('off')
plt.show()