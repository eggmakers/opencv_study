import cv2
import numpy as np

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

#LoG算子
def LoG(gray):
    #高斯滤波
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    #边缘检测
    dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize = 3)
    log = cv2.convertScaleAbs(dst)
    return log

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, ((int(img.shape[1]/2), int(img.shape[0]/2))), interpolation=cv2.INTER_AREA)
# img = sp_noise(img, 0.006)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
result = LoG(img_gray)

cv2.imshow("Origin", img_gray)
cv2.imshow("LoG", result)
cv2.waitKey()