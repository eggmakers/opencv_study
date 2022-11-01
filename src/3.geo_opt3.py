import imp


import math
import cv2
import numpy as np

x, y = 3, 5
print('直角坐标x = ',x, '\n 直角坐标y = ',y)
#mash库函数计算
center = [0, 0]                                                         #中心点
r = math.sqrt(math.pow(x - center[0], 2) + math.pow(y - center[1], 2))
theta = math.atan2(y - center[1], x - center[0]) / math.pi * 180        #转换为角度
print('math库r = ',r)
print('math库theta = ',theta)

#OpenCV也提供了及坐标变换的函数
x1 = np.array(x, np.float32)
y1 = np.array(y, np.float32)
#变换中心为原点，若想为（2，3）需x1-2,y1-3
r1, theta1 = cv2.cartToPolar(x1, y1, angleInDegrees = True)
print('OpenCV库函数r1 = ',r1)
print('OpenCV库函数theta1 = ',theta1)

#反变换
x1,y1 = cv2.polarToCart(r1, theta1, angleInDegrees = True)
print('极坐标变为笛卡尔坐标x = ',np.round(x1[0]))
print('极坐标变为笛卡尔坐标y = ',np.round(y1[0]))