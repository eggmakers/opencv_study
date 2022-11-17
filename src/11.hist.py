import cv2
import matplotlib.pyplot as plt

img = cv2.imread("picture_material/beauty_leg3.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)

plt.figure("full color histogram1")
hist0 = cv2.calcHist([img[:, :, 0]], [0], None, [256], [0, 256])
plt.plot(hist0, color = 'b')
hist1 = cv2.calcHist([img[:, :, 1]], [0], None, [256], [0, 256])
plt.plot(hist1, color = 'g')
hist2 = cv2.calcHist([img[:, :, 2]], [0], None, [256], [0, 256])
plt.plot(hist2, color = 'r')

img0_equ = cv2.equalizeHist(img[:,:,0])
img1_equ = cv2.equalizeHist(img[:,:,1])
img2_equ = cv2.equalizeHist(img[:,:,2])

image = cv2.merge(img0_equ, img1_equ, img2_equ)
plt.figure("full color histogram2")
hist_equ = cv2.calcHist([image], [0], None, [256], [0, 256])
plt.plot(hist_equ)

cv2.imshow("Histogram Equalization", image)
plt.show()
cv2.waitKey()