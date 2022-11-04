import cv2
import numpy as np
import matplotlib.pyplot as plt

def Origin_histogram(img):
    #建立原始图像个灰度级的灰度值与像素个数对应表
    histogram = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            k = img[i][j]
            if k in histogram:
                histogram[k] += 1
            else:
                histogram[k] = 1
    sorted_histogram = {}               
    sorted_list = sorted(histogram)
    for j in range(len(sorted_list)):
        sorted_histogram[sorted_list[j]] = histogram[sorted_list[j]]
    return sorted_histogram

def equalization_histogram(histogram, img):
    pr = {}
    for i in histogram.keys():
        pr[i] = histogram[i] / (img.shape[0] * img.shape[1])
    tmp = 0
    for m in pr.keys():
        tmp += pr[m]
        pr[m] = max(histogram) * tmp
    new_img = np.zeros(shape = (img.shape[0], img.shape[1]), dtype = np.uint8)
    
    for k in range(img.shape[0]):
        for l in range(img.shape[1]):
            new_img[k][l] = pr[img[k][l]]
    return new_img

def GrayHist(img):
    height, width = img.shape[0: 2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(height):
        for j in range(width):
            grayHist[img[i][j]] += 1
    return grayHist

if __name__ == '__main__':
    img = cv2.imread("picture_material/beauty_leg3.jpg")
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
    
    origin_histogram = Origin_histogram(img)
    #直方图均衡化
    new_img = equalization_histogram(origin_histogram, img)
    origin_grayHist = GrayHist(img)
    equaliza_grayHist = GrayHist(new_img)
    #绘制灰度直方图
    x = np.arange(256)
    plt.figure(num = 1)
    plt.plot(x, origin_grayHist, 'r', linewidth = 1, c = 'blue')
    plt.title("Origin"), plt.ylabel("number of pixels")
    
    plt.figure(num = 2)
    plt.plot(x, equaliza_grayHist, 'r', linewidth = 1, c = 'blue')
    plt.title("Equalization"), plt.ylabel("number of pixels")
    plt.show()
    
    cv2.imshow("Origin", img)
    cv2.imshow("Equalization", new_img)
    cv2.waitKey()
