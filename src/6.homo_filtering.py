import cv2
import numpy as np
import matplotlib.pyplot as plt

def homomorphic_filter(src, d0 = 10, r1 = 0.5, rh = 2, c = 4, h = 2.0, l = 0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape
    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt((M ** 2 + N ** 2))
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    image_filtering_fftshift = Z * gray_fftshift
    image_filtering_fftshift = (h - 1) * image_filtering_fftshift + 1
    image_filtering_ifftshift = np.fft.ifftshift(image_filtering_fftshift)
    image_filtering_ifft = np.fft.ifft2(image_filtering_ifftshift)
    image_filtering = np.real(image_filtering_ifft)
    image_filtering = np.uint8(np.clip(image_filtering, 0, 255))
    return image_filtering

if __name__ == '__main__':
    image = cv2.imread("picture_material/beauty_leg3.jpg")
    image_homomorphic_filter = homomorphic_filter(image)
    plt.figure('homo_filtering')
    plt.subplot(121), plt.axis('off')
    plt.imshow(image,'gray'), plt.title('Origin image')
    plt.subplot(122), plt.imshow(image_homomorphic_filter, 'gray')
    plt.title('Homomorphic image'), plt.axis('off')
    plt.show()