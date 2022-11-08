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

def inverse(input, PSF, eps):
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    result = np.fft.ifft2(input_fft / PSF_fft)
    result = np.abs(np.fft.fftshift(result))
    return result

if __name__ == '__main__':
    img = cv2.imread("picture_material/beauty_leg3.jpg")
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #进行运动模糊处理
    img_h,img_w = img.shape[:2]
    PSF = motion_process((img_h, img_w), 60)
    blurred = np.abs(make_blurred(img, PSF, 1e-3))
    plt.figure("inverse")
    plt.subplot(221),plt.axis('off')
    plt.title('motion blurred')
    plt.imshow(blurred)
    
    result_inv = inverse(blurred, PSF, 1e-3)
    plt.subplot(222),plt.axis('off')
    plt.title('result_inv')
    plt.imshow(result_inv)
    
    #添加噪声，standard_normal产生随机的函数
    blurred_noisy = blurred + 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)
    plt.subplot(223),plt.axis('off')
    plt.title('motion & noise blurred')
    plt.imshow(blurred_noisy)
    
    result_invn = inverse(blurred_noisy, PSF, 0.1 + 1e-3)
    plt.subplot(224),plt.axis('off')
    plt.title('Noise inverse deblurred')
    plt.imshow(result_invn)
    plt.show()
    