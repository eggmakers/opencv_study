# python_opencv学习

## 数字图形的获取和基本运算



|          值          |                             含义                             | 数值 |
| :------------------: | :----------------------------------------------------------: | :--: |
| cv2.IMREAD_UNCHANGED |                        保持原格式不变                        |  -1  |
| cv2.IMREAD_GRAYSCALE |                 将图像调整为单通道的灰度图像                 |  0   |
|   cv2.IMREAD_COLOR   |       将图像调整为3通道的BGR图像，该值为flags的默认值        |  1   |
| cv2.IMREAD_ANYDEPTH  | 当载入的图像深度为16位或者是32位时，就返回其对应的深度图像，否则，将其转换为8位图像 |  2   |
| cv2.IMREAD_ANYCOLOR  |                 以任何可能的颜色格式读取图像                 |  4   |
| cv2.IMREAD_LOAD_GDAL |                   使用GDAL驱动程序加载图像                   |  8   |

### 图像的算术运算

#### 加法计算

`cv2.add(src1,src2)#加法运算，src为图像矩阵`

产生离散均匀分布的整数函数：

`np.random.randint(low, high = None, size = None, dtype = '1')`

`#low:生成元素的最小值`

`#high:最大值`

`#size:输出的大小`

`#dtype:生成的元素类型`

`cv2.add()`

![add](F:/Users/14024/Desktop/python_opencv学习/result/result1 = cv2.add(img1,img2).png)

`a+b`

![a+b](F:/Users/14024/Desktop/python_opencv学习/result/result2 = img1 + img2.png)

图像融合算法如下：

`dst = src1 * alpha + src2 * beta +  gamma`

函数为：

`dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, type])`

dst为输出图像

src1为输入的第一幅图象

alpha为第一幅图象的权重

src2为矩阵数一样的第二幅图

beta为第二幅图象的权重

gamma为调亮度

dtype为可选深度，默认为-1

![a+b](F:/Users/14024/Desktop/python_opencv学习/result/result3 = cv2.addWeighted(img1+img2).png)

#### 减法计算

`dst = cv2.subtract(src1,src2)#减法计算`

![a+b](F:/Users/14024/Desktop/python_opencv学习/result/result5 = cv2.subtract(img1+img2).png)

a - b

![a+b](F:/Users/14024/Desktop/python_opencv学习/result/result4 = img1 - img2.png)

#### 乘法计算

`result =  np.dot(a, b)`

![a+b](F:/Users/14024/Desktop/python_opencv学习/result/result6 = cv2.multiply(img1).png)

#### 除法计算

`result = cv2.divide(a, b)`

![a+b](F:/Users/14024/Desktop/python_opencv学习/result/result7 = cv2.divide(img1).png)

### 图像的逻辑计算

#### 按位与计算

真值表

| 输入值a | 输入值b | 输出结果 |
| :-----: | :-----: | :------: |
|    0    |    0    |    0     |
|    0    |    1    |    0     |
|    1    |    0    |    0     |
|    1    |    1    |    1     |

`result = cv2.bitwise_and(src1,src2)`

<img src="F:/Users/14024/Desktop/python_opencv学习/result/img1.png" alt="a+b" style="zoom:33%;" />

<img src="F:/Users/14024/Desktop/python_opencv学习/result/img2.png" alt="a+b" style="zoom:33%;" />

![a+b](F:/Users/14024/Desktop/python_opencv学习/result/img3.png)

#### 按位或计算

真值表

| 输入a | 输入b | 输出结果 |
| :---: | :---: | :------: |
|   0   |   0   |    0     |
|   0   |   1   |    1     |
|   1   |   0   |    1     |
|   1   |   1   |    1     |

`result = cv2.bitwise_or(src1,src2)`

![a+b](F:/Users/14024/Desktop/python_opencv学习/result/img4.png)

#### 按位非运算

`result = cv2.bitwise_not(src)`

| 输入 | 输出 |
| :--: | :--: |
|  0   |  1   |
|  1   |  0   |



![a+b](F:/Users/14024/Desktop/python_opencv学习/result/img5.png)

#### 按位异或运算

`result = cv2.bitwise_xor(src1,src2)`

| 输入值a | 输入值b | 输出结果 |
| :-----: | :-----: | :------: |
|    0    |    0    |    0     |
|    0    |    1    |    1     |
|    1    |    0    |    1     |
|    1    |    1    |    0     |

![a+b](F:/Users/14024/Desktop/python_opencv学习/result/img6.png)

#### 综合示例

```python
#在图像上添加一个图像
import cv2

img1 = cv2.imread("picture_material/batman.jpg")
img1 = cv2.resize(img1, (int(img1.shape[1]/2),int(img1.shape[0]/2)), interpolation=cv2.INTER_AREA)
img2 = cv2.imread("picture_material/logo.jpg")
img2 = cv2.resize(img2, (int(img2.shape[1]/4),int(img2.shape[0]/4)), interpolation=cv2.INTER_AREA)

#创建ROI
rows1,cols1,channels1 = img1.shape
rows,cols,channels = img2.shape
roi = img1[0:rows, (cols1-cols):cols1]

#创建logo掩码，并同时创建相反掩码
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(img2gray, 240, 255, cv2.THRESH_BINARY)
#mask背景是白色，彩色logo是黑色
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi, roi, mask = mask)
img2_bg = cv2.bitwise_and(img2,img2, mask = mask_inv)

#将logo放入ROI，并修正主图像
dst = cv2.add(img1_bg,img2_bg)
img1[0:rows, (cols1 - cols):cols1] = dst

cv2.imshow("Result",img1)
cv2.waitKey()
```

![a+b](F:/Users/14024/Desktop/python_opencv学习/result/result.png)

#### 练习见代码

## 数字图像的几何运算

### 图像平移

`dst = cv2.warpAffine(src,M,dsize[,flags[,borderMode[,borderValue]]])`

src为输入图像

M为变换矩阵，反映平移或旋转

dsize为输出图像大小

flags为插值方法（见下表）

borderMode为边界像素模式

borderValue为边界像素填充值

|             类型             |                     说明                     |
| :--------------------------: | :------------------------------------------: |
|      cv2.INTER_NEAREST       |                  最近邻插值                  |
|       cv2.INTER_LINEAR       |              双线性插值（默认）              |
|       cv2.INTER_CUBIC        |                 三次样条插值                 |
|        cv2.INTER_AREA        | 区域插值，根据周边像素值实现当前像素点的采样 |
|    cv2.INTER_LINEAR_EXACT    |               位精确双线性插值               |
|        cv2.INTER_MAX         |                 插值编码掩码                 |
| cv2.INTER_WARP_FILL_OUTLIERS |              标志，填补所有像素              |
|     cv2.WARP_INVERSE_MAP     |                    逆变换                    |

转换矩阵如下：

向上平移：(把它想象成一个矩阵)

|  1   |  0   | t_x  |
| :--: | :--: | :--: |
|  0   |  1   | t_y  |

<img src="F:/Users/14024/Desktop/python_opencv学习/result/New_picture.png" alt="a+b" style="zoom:33%;" />

<img src="F:/Users/14024/Desktop/python_opencv学习/result/New_picture1.png" alt="a+b" style="zoom:33%;" />

### 仿射变换的类型

`M = cv2.getAffineTransform(src, dst)`

src为原始图像的三点坐标

dst为变换三点的坐标

M为仿射变换矩阵

![a+b](F:/Users/14024/Desktop/python_opencv学习/result/Figure_1.png)

### 图像缩放

`dst = cv2.resize(src, dsize[,fx[,fy[,interpolation]]])`

这个就不说了，几乎每个程序都用，快背下来了

### 图像旋转

`M = cv2.getRotationMatrix2D(center, angle, scale)`

center:旋转中心

angle:旋转角度

scale:缩放比例以及旋转方向，正数为逆时针，负数为顺时针

<img src="F:/Users/14024/Desktop/python_opencv学习/result/Image rotation1.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/python_opencv学习/result/Image rotation2.png" alt="a+b" style="zoom:50%;" />

### 图像剪切

#### 规则剪切

按矩形进行剪切

`img = img[a:b, c:d]`

#### 不规则剪切

见程序

```python
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
```

### 图像的镜像变换

`dst = cv2.flip(src, flipCode)`

src为输入图像

flipCode为翻转方向，0表示绕x轴旋转，1表示绕y轴旋转，-1表示绕x轴，y轴两个轴旋转，即中心旋转

![a+b](F:/Users/14024/Desktop/python_opencv学习/result/Figure_2.png)

### 图像的透视变换

`M = cv2.getPerspectiveTransform(src, dst[, solveMethod])`

透视变换就是将图片投影到一个新的视平面，也称为投影映射。

其中dst为透视变换后的4个对应点的位置

M为生成的透视变换矩阵

透视变换函数为

`dst = cv2.warpPerspective(src, M, dsize[, flags[, borderMode[, borderValue]]])`

dsize为输出图像的尺寸

![a+b](F:/Users/14024/Desktop/python_opencv学习/result/Figure_3.png)

### 图像的极坐标变换

#### 数据点坐标系间的转换

