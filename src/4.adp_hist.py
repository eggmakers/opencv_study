import cv2
import matplotlib.pyplot as plt

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#创建CLAHE对象
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8, 8))
dst = clahe.apply(img)

plt.figure("原始直方图")
plt.hist(img.ravel(), 256)
plt.figure("自适应直方图均衡化")
plt.hist(img.ravel(), 256)
plt.show()

cv2.imshow('Origin', img)
cv2.imshow('CLAHE', dst)
cv2.waitKey()