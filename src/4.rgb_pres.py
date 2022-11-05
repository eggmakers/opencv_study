import cv2
import numpy as np

img1 = cv2.imread("picture_material/office.jpg")
img2 = cv2.imread("picture_material/beauty_leg3.jpg")
img2 = cv2.resize(img2, (int(img2.shape[1]/2), int(img2.shape[0]/2)), interpolation=cv2.INTER_AREA)
cv2.imshow('Origin', img1)
cv2.imshow('Target', img2)

color = ('r', 'g', 'b')
for i, col in enumerate(color):
    hist1, bins = np.histogram(img1[:, :, i].ravel(), 256, [0, 256])
    hist2, bins = np.histogram(img2[:, :, i].ravel(), 256, [0, 256])
    cdf1 = hist1.cumsum()       #累计计算数组
    cdf2 = hist2.cumsum()
    cdf1_hist = hist1.cumsum() / cdf1.max() #灰度值的累计值的比率
    cdf2_hist = hist2.cumsum() / cdf2.max()
    
    diff_cdf = [[0 for j in range(256)] for k in range(256)]
    
    for j in range(256):
        for k in range(256):
            diff_cdf[j][k] = abs(cdf1_hist[j] - cdf2_hist[k])
            
    lut = [0 for j in range(256)]
    for j in range(256):
        min = diff_cdf[j][0]
        index = 0
        for k in range(256):
            if min > diff_cdf[j][k]:
                min = diff_cdf[j][k]
                index = k
        lut[j] = ([j, index])
        
    h = int(img1.shape[0])
    w = int(img1.shape[1])
    
    for j in range(h):
        for k in range(w):
            img1[j, k, i] = lut[img1[j, k, i]][1]
            
img3 = img1
cv2.imshow('Specification Img', img3)
cv2.waitKey()