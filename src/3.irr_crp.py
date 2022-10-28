import cv2
import numpy as np

def On_Mouse(event, x, y, flags, param):
    global img, point1, point2, count, pointsMax
    global IsPointsChoose, tpPointsChoose           #存入选择的点
    global pointsCount                              #对鼠标按下的点计数
    global img2, ROI_bymouse_flag
    img2 = img.copy()                               #保证每次都重新在画原画
    
    if event == cv2.EVENT_LBUTTONDOWN:
        pointsCount = pointsCount + 1
        print('pointsCount:',pointsCount)
        point1 = (x, y)
        print(x, y)
        cv2.circle(img2, point1, 5, (0, 255, 0), 2) #画出单击的点
        
        #将选取的点保存到list列表里
        IsPointsChoose.append([x, y])               #用于转化为darry提取多边形
        tpPointsChoose.append([x, y])               #用于画点
        #将鼠标选的点用直线连接起来
        print(len(tpPointsChoose))
        for i in range(len(tpPointsChoose) - 1):
            print('i',i)
            cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
        cv2.imshow('src',img2)
    
#---右键点击，清除轨迹-----------
    if event == cv2.EVENT_RBUTTONDOWN:
        print("right-mouse")
        pointsCount = 0
        tpPointsChoose = []
        IsPointsChoose = []
        print(len(tpPointsChoose))
        for i in range(len(tpPointsChoose) - 1):
            print('i', i)
            cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
        cv2.imshow('src',img2)
        
#---双击鼠标，结束选取，绘制感兴趣区域----
    if event == cv2.EVENT_LBUTTONDBLCLK:
        ROI_byMouse()

def ROI_byMouse():
    global src, ROI, ROI_flag, mask2
    mask = np.zeros(img.shape, np.uint8)
    pts = np.array([IsPointsChoose], np.int32)                  #pts是多边形的顶点列表
    pts = pts.reshape((-1, 1, 2))
    #OpenMV中需要先将多边形的顶点坐标变成顶点数 X1 X2维的矩阵，再进行绘制
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255))    #画多边形
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))          #填充多边形
    cv2.imshow('mask', mask2)
    #掩模图像与原图像进行“位与”操作
    ROI = cv2.bitwise_and(mask2, img) 
    cv2.imshow('ROI', ROI)
    
if __name__ == '__main__':
    #选择点设置
    IsPointsChoose = []
    tpPointsChoose = []
    pointsCount = 0
    count = 0
    pointsMax = 6
    
    img = cv2.imread("picture_material/beauty_leg3.jpg")
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
    ROI = img.copy()
    cv2.namedWindow('src')
    cv2.setMouseCallback('src', On_Mouse)
    cv2.imshow('src', img)
    cv2.waitKey()
    
cv2.destroyAllWindows()