import cv2
import numpy as np

#计算灰度
def calcGrayHist(grayImage):
    rows, cols = grayImage.shape
    grayHist = np.zeros([256], np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[grayImage[r][c]] += 1
    return grayHist

#直方图阈值分割
def threshTwoPeaks(image):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    #计算灰度直方图
    histogram = calcGrayHist(gray)
    maxLoc = np.where(histogram == np.max(histogram))
    firstPeak = maxLoc[0][0]
    #寻找第二个峰值
    measureDists = np.zeros([256], np.float32)
    for k in range(256):
        measureDists[k] = pow(k - firstPeak, 2) * histogram[k]
    maxLoc2 = np.where(measureDists == np.max(measureDists))
    secondPeak = maxLoc2[0][0]
    
    thresh = 0
    if firstPeak > secondPeak:
        temp = histogram[int(secondPeak) : int(firstPeak)]
        minloc = np.where(temp == np.min(temp))
        thresh = secondPeak + minloc[0][0] + 1
    else:
        temp = histogram[int(firstPeak) : int(secondPeak)]
        minloc = np.where(temp == np.min(temp))
        thresh = firstPeak + minloc[0][0] + 1
        
    #图像二值化
    threshImage_out = gray.copy()
    threshImage_out[threshImage_out > thresh] = 255
    threshImage_out[threshImage_out <= thresh] = 0
    return thresh, threshImage_out

if __name__ == '__main__':
    img = cv2.imread("picture_material/beauty_leg2.jpg")
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
    thresh, out_img = threshTwoPeaks(img)
    print("thresh = ", thresh)
    cv2.imshow("origin", img)
    cv2.imshow("hist_edge", out_img)
    cv2.waitKey()
    