import cv2
import matplotlib.pyplot as plt

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(img_Gray)

plt.figure("原始灰度直方图")
plt.title("Origin")
plt.hist(img_Gray.ravel(), 256)

plt.figure("均衡化直方图")
plt.title("Equalization")
plt.hist(equ.ravel(), 256)
plt.show()

cv2.imshow("Gray", img_Gray)
cv2.imshow("EqualizeHist", equ)
cv2.waitKey()