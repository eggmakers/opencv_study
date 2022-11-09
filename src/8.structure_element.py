import cv2
import numpy as np

#矩形内核
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
print(kernel1)

#椭圆内核
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
print(kernel2)

#十字内核
kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
print(kernel3)

#正方形内核
kernel4 = np.ones((5, 5), np.uint8)
print(kernel4)

#菱形内核
kernel5 = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=np.uint8)
print(kernel5)