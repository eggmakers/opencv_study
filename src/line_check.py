import cv2
import numpy as np

img = cv2.imread("picture_material/road.jpg")
img = cv2.resize(img, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
img = cv2.GaussianBlur(img, (3, 3), 0)
edges = cv2.Canny(img, 50, 150, apertureSize = 3)
lines = cv2.HoughLines(edges, 1, np.pi/180, 110)
print("line num : ", len(lines))

result = img.copy()
for line in lines:
    rho = line[0][0]
    theta = line[0][1]
    if (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)):
        pt1 = (int(rho/np.cos(theta)), 0)
        pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
        cv2.line(result, pt1, pt2, (0, 255, 0))
    else:
        pt1 = (0, int(rho/np.sin(theta)))
        pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
        cv2.line(result, pt1, pt2, (0, 0, 255), 1)
    
cv2.imshow('Origin', img)
cv2.imshow('Canny', edges)
cv2.imshow('Result', result)    
        
#霍夫变换
minLineLength = 500
maxLineGap = 40
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength, maxLineGap)
print("line num : ", len(lines))

#画出检测的线段
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
cv2.imshow('Result', img)
cv2.waitKey()