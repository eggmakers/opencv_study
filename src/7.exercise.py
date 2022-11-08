#了解图像退化与复原的机理，根据运动模糊的模型函数生成任意角度的运动模糊图像
import numpy as np
import cv2
 
def motion_blur(image, degree=12, angle=50):
    image = np.array(image)
 
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
 
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred
 
img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
img_blurred = motion_blur(img)
 
cv2.imshow('Source image',img)
cv2.imshow('blur image',img_blurred)
cv2.waitKey()

#编写程序，创建运动模糊核矩阵，利用该矩阵生成运动模糊图像
#就是上式
#编写程序，使用逆滤波算法对高斯噪声图像进行滤波
import math
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

def gasuss_noise(img, mean=0, var=0.01):
    #高斯噪声，mean：均值；var：方差
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    img_noise = img + noise
    if img_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    img_noise = np.clip(img_noise, low_clip, 1.0)
    img_noise = np.uint8(img_noise * 255)
    return img_noise

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
    blurred_noisy = gasuss_noise(img)
    plt.subplot(223),plt.axis('off')
    plt.title('motion & noise blurred')
    plt.imshow(blurred_noisy)
    
    result_invn = inverse(blurred_noisy, PSF, 0.1 + 1e-3)
    plt.subplot(224),plt.axis('off')
    plt.title('Noise inverse deblurred')
    plt.imshow(result_invn)
    plt.show()

#编写程序，分别使用逆滤波和维纳滤波对乘性噪声图像进行滤波处理，比较它们的滤波效果
def mul_noise(o_img, standard_deviation):
    gaussian_noise = np.random.normal(loc=0, scale=standard_deviation, size=o_img.shape)
    r = o_img.shape[0]
    c = o_img.shape[1]
    noisy_img = np.zeros((r, c), dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            # apply noise for every pixel
            noise = o_img[i, j] * (1 + gaussian_noise[i, j])
            if noise < 0:
                noise = 0
            elif noise > 255:
                noise = 255
            noisy_img[i, j] = noise
    # return the result image
    return noisy_img

def wiener(input, PSF, eps, K = 0.01):
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = np.fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(np.fft.fftshift(result))
    return result

if __name__ == '__main__':
    img = cv2.imread("picture_material/beauty_leg3.jpg")
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_spk = mul_noise(img, 0.1)
    img_h,img_w = img.shape[:2]
    PSF = motion_process((img_h, img_w), 60)
    cv2.imshow("speckle", img_spk)
    img_inverse_mul = inverse(img_spk, PSF, 1e-3)
    cv2.imshow("img_inverse_mul", img_inverse_mul)
    img_wiener_mul = inverse(img_spk, PSF, 1e-3)
    cv2.imshow("img_wiener_mul", img_wiener_mul)
    cv2.waitKey()