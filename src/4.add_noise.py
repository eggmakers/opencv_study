import cv2
import numpy as np
import math

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

def random_noise(img, noise_num):
    #随机噪声 noise_num:噪声点数目
    img_noise = img
    rows, cols, chn = img_noise.shape
    for i in range(noise_num):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    return img_noise

def poisson_noise(img):
    #泊松噪声
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img * vals) / float(vals)
    return noisy

def mul_noise(o_img, standard_deviation):
    #乘性噪声
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

def rayleigh_noise(img):
    #瑞利噪声
    a = -0.2
    b = 0.03
    row,col,ch = img.shape
    n_reyleigh = a + (-b * math.log(1 - np.random.randn(row, col))) ** 0.5
    return n_reyleigh

def Gamma_noise(img):
    #伽马噪声
    a = 25
    b = 3
    row,col,ch = img.shape
    n_gamma = np.zeros(row, col)
    for i in range(b):
        n_gamma = n_gamma + (-1 / a) * math.log(1 - np.random.randn(row, col))
    return n_gamma

img = cv2.imread("picture_material/beauty_leg3.jpg")
img = cv2.resize(img, ((int(img.shape[1]/2), int(img.shape[0]/2))), interpolation=cv2.INTER_AREA)
cv2.imshow("origin", img)

img_gasuss = gasuss_noise(img, mean = 0, var = 0.01)
cv2.imshow("gasuss", img_gasuss)

img_sp = sp_noise(img, 0.06)
cv2.imshow("sp", img_sp)

img_random_noise = random_noise(img, 1000)
cv2.imshow("random", img_random_noise)

cv2.waitKey()