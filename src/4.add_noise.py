import cv2
import numpy as np

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
