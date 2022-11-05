import cv2
import numpy as np
import matplotlib.pyplot as plt

#定义计算直方图累积概率函数
def histCalculate(src):
    row, col = np.shape(src)
    hist = np.zeros(256, dtype = np.float32)
    cumhist = np.zeros(256, dtype = np.float32)
    cumProbhist = np.zeros(256, dtype = np.float32)
    #y轴归一化
    for i in range(row):
        for j in range(col):
            hist[src[i][j]] += 1
            
    cumhist[0] = hist[0]
    for i in range(1, 256):
        cumhist[i] = cumhist[i - 1] + hist[i]
    cumProbhist = cumhist / (row * col)
    return cumProbhist

#定义实现直方图规定化函数
def histSpecification(specImg, refeImg):
    spechist = histCalculate(specImg)
    refehist = histCalculate(refeImg)
    corspdValue = np.zeros(256, dtype = np.uint8)
    #直方图规定化
    for i in range(256):
        diff = np.abs(spechist[i] - refehist[i])
        matchValue = i
        for j in range(256):
            if np.abs(spechist[i] - refehist[i]) < diff:
                diff = np.abs(spechist[i] - refehist[j])
                matchValue = j
        corspdValue[i] = matchValue
    outputImg = cv2.LUT(specImg, corspdValue)
    return outputImg

#原图像
img1 = cv2.imread("picture_material/office.jpg", cv2.IMREAD_GRAYSCALE)
#参考图
img2 = cv2.imread("picture_material/beauty_leg3.jpg", cv2.IMREAD_GRAYSCALE)

# img1 = cv2.resize(img1, (int(img1.shape[1]/2), int(img1.shape[0]/2)), interpolation = cv2.INTER_AREA)
img2 = cv2.resize(img2, (int(img2.shape[1]/2), int(img2.shape[0]/2)), interpolation = cv2.INTER_AREA)

cv2.imshow("Input Img",img1)
cv2.imshow("Reference Img", img2)
imgOutput = histSpecification(img1, img2)
cv2.imshow("Output Img",imgOutput)
cv2.waitKey()