import cv2
import math

img = cv2.imread("picture_material/1.jpg")
h,w = img.shape[0:2]
maxRadius = math.hypot(w / 2, h / 2)
m = w / math.log(maxRadius)
log_polar = cv2.logPolar(img, (w / 2, h / 2), m, cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)
linear_polar = cv2.linearPolar(img, (w / 2, h / 2), m, cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)

log_dst = cv2.transpose(log_polar)      #图像转置
log_dst = cv2.flip(log_dst, 0)          #图像垂直镜像
lin_dst = cv2.transpose(linear_polar)   #图像转置
lin_dst = cv2.flip(lin_dst, 0)          #图像垂直镜像

cv2.imshow("Original",img)
cv2.imshow('log_polar',log_dst)
cv2.imshow('linear_polar',lin_dst)

cv2.waitKey()
cv2.destroyAllWindows()