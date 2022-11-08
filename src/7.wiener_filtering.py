import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

#运动模糊
def motion_process(img_size, motion_angle):
    PSF = np.zeros(img_size)
    center_position = (img_size[0] - 1) / 2
    
    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(15):
            offset = round(i * slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF /PSF.sum()
    else:
        for i in range(15):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF /PSF.sum()
    
def make_blurred(input, PSF, eps):
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    blurred = np.fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(np.fft.fftshift(blurred))
    return blurred

def wiener(input, PSF, eps, K = 0.01):
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = np.fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(np.fft.fftshift(result))
    return result

if __name__ == '__main__':
    img = cv2.imread("picture_material/beauty_leg2.jpg")
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img.shape[:2]
    PSF = motion_process((img_h, img_w), 60)
    blurred = np.abs(make_blurred(img, PSF, 1e-3))
    plt.figure("wiener")
    plt.subplot(221), plt.axis('off')
    plt.title('Motion blurred')
    
    resultwd = wiener(blurred, PSF, 1e-3)
    plt.subplot(222), plt.axis('off')
    plt.title('wiener deblurred(k=0.01)')
    plt.imshow(resultwd)
    
    #添加随机噪声
    blurred_noisy = blurred + 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)
    plt.subplot(223), plt.axis('off')
    plt.title('motion & noise blurred')
    plt.imshow(blurred_noisy)
    
    resultwdn = wiener(blurred_noisy, PSF, 1e-3)
    plt.subplot(224), plt.axis('off')
    plt.title('wiener deblurred(k=0.01)')
    plt.imshow(resultwdn)
    plt.show()