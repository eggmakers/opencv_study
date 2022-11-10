import cv2

def nothing(*args):
    pass

cv2.namedWindow('morphology', cv2.WINDOW_FREERATIO)
r, MAX_R = 0, 20#初始半径
i, MAX_I = 0, 20#初始化迭代次数

#创建滑动条，分别为半径和迭代次数
cv2.createTrackbar('r', 'morphology', r, MAX_R, nothing)
cv2.createTrackbar('i', 'morphology', i, MAX_I, nothing)

src = cv2.imread("picture_material/beauty_leg2.jpg")
while True:
    r = cv2.getTrackbarPos('r', 'morphology')#获得进度条上的r值
    i = cv2.getTrackbarPos('i', 'morphology')
    #创建结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    #形态梯度
    result = cv2.morphologyEx(src, cv2.MORPH_GRADIENT, kernel, iterations = i)
    #显示效果
    cv2.imshow('morphology', result)
    
    ch = cv2.waitKey(1)
    if ch == 27:
        break
    
cv2.destroyAllWindows()