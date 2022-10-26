#使用numpy函数生成两个矩阵
import numpy as np
import cv2
#定义两个随机矩阵
img1 = np.random.randint(0, 256, size = [4,4], dtype = np.uint8)
img2 = np.random.randint(0, 256, size = [4,4], dtype = np.uint8)
img3 = cv2.add(img1, img2)
img4 = cv2.subtract(img1, img2)
img5 = np.dot(img1, img2)
img6 = cv2.multiply(img1, img2)
img7 = cv2.divide(img1, img2)

print("img1 = \n",img1)
print("img2 = \n",img2)
print("result = img1 + img2 \n", img1 + img2)
print("result = img1 + img2 \n", img3)
print("result = img1 - img2 \n", img1 - img2)
print("result = img1 - img2 \n", img4)
print("result = img1 * img2 \n", img5)
print("result = img1 * img2 \n", img6)#结果大于255时，变成255
print("result = img1 / img2 \n", img7)#自动取整