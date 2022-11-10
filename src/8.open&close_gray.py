import cv2

#设置回调函数
def nothing(*args):
    pass

img = cv2.imread("picture_material/beauty_leg2.jpg")
cv2.namedWindow('morphology_open', 6)
cv2.namedWindow('morphology_close', 6)
r1, MAX_R1 = 1, 20
i1 ,MAX_I1 = 1, 20
r2, MAX_R2 = 1, 20
i2 ,MAX_I2 = 1, 20

cv2.createTrackbar('r', 'morphology_open', r1, MAX_R1, nothing)
cv2.createTrackbar('i', 'morphology_open', i1, MAX_I1, nothing)
cv2.createTrackbar('r', 'morphology_close', r2, MAX_R2, nothing)
cv2.createTrackbar('i', 'morphology_close', i2, MAX_I2, nothing)

while True:
    r1 = cv2.getTrackbarPos('r', 'morphology_open')
    i1 = cv2.getTrackbarPos('i', 'morphology_open')
    r2 = cv2.getTrackbarPos('r', 'morphology_close')
    i2 = cv2.getTrackbarPos('i', 'morphology_close')
    #创建卷积核
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r1 + 1, 2 * r1 + 1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r2 + 1, 2 * r2 + 1))
    #开运算
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1, iterations = i1)
    cv2.imshow('morphology_open', result)#显示效果
    #闭运算
    result1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2, iterations = i2)
    cv2.imshow('morphology_close', result1)
    
    ch = cv2.waitKey(1)
    if ch == 27:
        break
cv2.destroyAllWindows()