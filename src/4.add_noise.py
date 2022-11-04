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

def sp_noise(img, prob):
    #椒盐噪声 prob：噪声比例
    img_noise = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rNum = np.random.random()
            if rNum < prob:
                img_noise[i][j] = 0
            elif rNum > thres:
                img_noise[i][j] = 255
            else:
                img_noise[i][j] = img[i][j]
    return img_noise

