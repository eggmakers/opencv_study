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

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, ((int(img.shape[1]/2), int(img.shape[0]/2))), interpolation=cv2.INTER_AREA)
# img = sp_noise(img, 0.006)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaus = cv2.GaussianBlur(img_gray, (3, 3), 0)
#Roberts算子
kernelx = np.array([[-1, 0], [0, 1]], dtype = int)
kernely = np.array([[0, -1], [1, 0]], dtype = int)
x = cv2.filter2D(img_gray, cv2.CV_16S, kernelx)
y = cv2.filter2D(img_gray, cv2.CV_16S, kernely)
#转uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#Prewitt算子
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype = int)
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype = int)
x = cv2.filter2D(img_gray, cv2.CV_16S, kernelx)
y = cv2.filter2D(img_gray, cv2.CV_16S, kernely)
#转uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#Sobel算子
sobelX = cv2.Sobel(img, cv2.CV_16S, 1, 0)#对x求一阶导
sobelY = cv2.Sobel(img, cv2.CV_16S, 0, 1)#对y求一阶导
#转uint8
sobelX = cv2.convertScaleAbs(sobelX)
sobelY = cv2.convertScaleAbs(sobelY)
sobelCombined = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
#拉普拉斯算子
laplacian1 = cv2.Laplacian(img, cv2.CV_16S, ksize = 1)
laplacian3 = cv2.Laplacian(img, cv2.CV_16S, ksize = 3)
laplacian5 = cv2.Laplacian(img, cv2.CV_16S, ksize = 5)
laplacian1 = cv2.convertScaleAbs(laplacian1)
laplacian3 = cv2.convertScaleAbs(laplacian3)
laplacian5 = cv2.convertScaleAbs(laplacian5)
#Canny算子
edge_output = cv2.Canny(img_gaus, 100, 200)
dst = cv2.bitwise_and(img, img, mask = edge_output)
#Scharr算子
x = cv2.Scharr(img, cv2.CV_16S, 1, 0)#x方向
y = cv2.Scharr(img, cv2.CV_16S, 0, 1)#y方向
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Scharr = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("Origin", img_gray)
cv2.imshow("Roberts", Roberts)
cv2.imshow("Prewitt", Prewitt)
cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY)
cv2.imshow("Sobel Combined", sobelCombined)
cv2.imshow("Laplacian1", laplacian1)
cv2.imshow("Laplacian3", laplacian3)
cv2.imshow("Laplacian5", laplacian5)
cv2.imshow("canny", edge_output)
cv2.imshow("Color dst", dst)
cv2.imshow("Scharr X", absX)
cv2.imshow("Scharr Y", absY)
cv2.imshow("Scharr", Scharr)
key = cv2.waitKey()

if key == 27:
    cv2.destroyAllWindows()
