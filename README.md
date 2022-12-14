# opencv_study

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

![add](F:/Users/14024/Desktop/opencv_study/result/result1 = cv2.add(img1,img2).png)

`a+b`

![a+b](F:/Users/14024/Desktop/opencv_study/result/result2 = img1 + img2.png)

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

![a+b](F:/Users/14024/Desktop/opencv_study/result/result3 = cv2.addWeighted(img1+img2).png)

#### 减法计算

`dst = cv2.subtract(src1,src2)#减法计算`

![a+b](F:/Users/14024/Desktop/opencv_study/result/result5 = cv2.subtract(img1+img2).png)

a - b

![a+b](F:/Users/14024/Desktop/opencv_study/result/result4 = img1 - img2.png)

#### 乘法计算

`result =  np.dot(a, b)`

![a+b](F:/Users/14024/Desktop/opencv_study/result/result6 = cv2.multiply(img1).png)

#### 除法计算

`result = cv2.divide(a, b)`

![a+b](F:/Users/14024/Desktop/opencv_study/result/result7 = cv2.divide(img1).png)

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

<img src="F:/Users/14024/Desktop/opencv_study/result/img1.png" alt="a+b" style="zoom:33%;" />

<img src="F:/Users/14024/Desktop/opencv_study/result/img2.png" alt="a+b" style="zoom:33%;" />

![a+b](F:/Users/14024/Desktop/opencv_study/result/img3.png)

#### 按位或计算

真值表

| 输入a | 输入b | 输出结果 |
| :---: | :---: | :------: |
|   0   |   0   |    0     |
|   0   |   1   |    1     |
|   1   |   0   |    1     |
|   1   |   1   |    1     |

`result = cv2.bitwise_or(src1,src2)`

![a+b](F:/Users/14024/Desktop/opencv_study/result/img4.png)

#### 按位非运算

`result = cv2.bitwise_not(src)`

| 输入 | 输出 |
| :--: | :--: |
|  0   |  1   |
|  1   |  0   |



![a+b](F:/Users/14024/Desktop/opencv_study/result/img5.png)

#### 按位异或运算

`result = cv2.bitwise_xor(src1,src2)`

| 输入值a | 输入值b | 输出结果 |
| :-----: | :-----: | :------: |
|    0    |    0    |    0     |
|    0    |    1    |    1     |
|    1    |    0    |    1     |
|    1    |    1    |    0     |

![a+b](F:/Users/14024/Desktop/opencv_study/result/img6.png)

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

![a+b](F:/Users/14024/Desktop/opencv_study/result/result.png)

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

<img src="F:/Users/14024/Desktop/opencv_study/result/New_picture.png" alt="a+b" style="zoom:33%;" />

<img src="F:/Users/14024/Desktop/opencv_study/result/New_picture1.png" alt="a+b" style="zoom:33%;" />

### 仿射变换的类型

`M = cv2.getAffineTransform(src, dst)`

src为原始图像的三点坐标

dst为变换三点的坐标

M为仿射变换矩阵

![a+b](F:/Users/14024/Desktop/opencv_study/result/Figure_1.png)

### 图像缩放

`dst = cv2.resize(src, dsize[,fx[,fy[,interpolation]]])`

这个就不说了，几乎每个程序都用，快背下来了

### 图像旋转

`M = cv2.getRotationMatrix2D(center, angle, scale)`

center:旋转中心

angle:旋转角度

scale:缩放比例以及旋转方向，正数为逆时针，负数为顺时针

<img src="F:/Users/14024/Desktop/opencv_study/result/Image rotation1.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Image rotation2.png" alt="a+b" style="zoom:50%;" />

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

![a+b](F:/Users/14024/Desktop/opencv_study/result/Figure_2.png)

### 图像的透视变换

`M = cv2.getPerspectiveTransform(src, dst[, solveMethod])`

透视变换就是将图片投影到一个新的视平面，也称为投影映射。

其中dst为透视变换后的4个对应点的位置

M为生成的透视变换矩阵

透视变换函数为

`dst = cv2.warpPerspective(src, M, dsize[, flags[, borderMode[, borderValue]]])`

dsize为输出图像的尺寸

![a+b](F:/Users/14024/Desktop/opencv_study/result/Figure_3.png)

### 图像的极坐标变换

#### 数据点坐标系间的转换

```python
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
```

#### 图像数据坐标系间的转换

`dst = cv2.LogPolar(src, center, M, int flags = CV2_INTER_LINEAR + CV2_WARP_FILL_OUTLIERS)`

center:直角坐标变换时直角坐标的原点坐标

M:幅度比例参数

flags:插值方法（CV_WARP_FILL_OUTLIERS表示填充所有目标图像像素）

![a+b](F:/Users/14024/Desktop/opencv_study/result/log_polar.png)

[^对数极坐标系5*5]: 对数变换

![a+b](F:/Users/14024/Desktop/opencv_study/result/linear_polar.png)

[^线性极坐标系的5*5]: 线性变换

#### 视频图像坐标系间的转换

`dst = cv2.warpPolar(src, dsize, center, maxRadius, flags)`

dsize:输出图像的尺寸

center:极坐标的原点坐标

maxRadius:变换时边界圆的半径

flags:插值法

|     标志参数      |       作用       |
| :---------------: | :--------------: |
| WARP_POLAR_LINEAR |    极坐标变换    |
|  WARP_POLAR_LOG   | 半对数极坐标变换 |
| WARP_INVERSE_MAP  |      逆变换      |

运行有问题，不打算查清

#### 练习见代码

## 图像空域增强

### 灰度线性变换

|   k   |    b    |     图像变换     |
| :---: | :-----: | :--------------: |
|   1   |    0    |     原始图像     |
|   1   | 不等于0 | 灰度值增加或降低 |
|  -1   |   255   |    灰度值反转    |
|  >1   |    -    |    对比度增强    |
| [0,1] |    -    |    对比度削弱    |
|  <0   |    -    |     图像求补     |



#### 用OPENCV做灰度变换与颜色空间变换

![a+b](F:/Users/14024/Desktop/opencv_study/result/Figure_4.png)

#### 增加或降低图像亮度

<img src="F:/Users/14024/Desktop/opencv_study/result/Origin Image.png" alt="a+b" style="zoom: 25%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/UP Gray.png" alt="a+b" style="zoom: 25%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Down_Gray.png" alt="a+b" style="zoom: 25%;" />

#### 增强或减弱图像对比度

<img src="F:/Users/14024/Desktop/opencv_study/result/Gray.png" alt="a+b" style="zoom: 25%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Enhancement.png" alt="a+b" style="zoom: 25%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Reduction.png" alt="a+b" style="zoom: 25%;" />

#### 图像反色变换

<img src="F:/Users/14024/Desktop/opencv_study/result/Origin Image.png" alt="a+b" style="zoom: 25%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Color_change.png" alt="a+b" style="zoom: 25%;" />

彩色图像反色

<img src="F:/Users/14024/Desktop/opencv_study/result/RGB.png" alt="a+b" style="zoom: 25%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Color_change_RGB.png" alt="a+b" style="zoom: 25%;" />

### 非线性变换

#### 对数变换

$$
g(x,y)=c*log(1+f(x,y))
$$

<img src="F:/Users/14024/Desktop/opencv_study/result/Figure_5.png" alt="a+b" style="zoom:25%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Origin.png" alt="a+b" style="zoom:25%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/LOG CHG.png" alt="a+b" style="zoom:25%;" />

#### 伽马变换

$$
g(x,y)=c*f(x,y)^r
$$

|  r   |                  图像变换                  |
| :--: | :----------------------------------------: |
|  >1  | 拉伸灰度级较高的区域，压缩灰度级较低的部分 |
|  <1  | 拉伸灰度级较低的区域，压缩灰度级较高的部分 |
|  =1  |                  线性变换                  |

<img src="F:/Users/14024/Desktop/opencv_study/result/Figure_6.png" alt="a+b" style="zoom:25%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Figure_7.png" alt="a+b" style="zoom:25%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Origin1.png" alt="a+b" style="zoom:25%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Gamma 》1.png" alt="a+b" style="zoom:25%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Gamma 《 1.png" alt="a+b" style="zoom:25%;" />

### 图像噪声

#### 通过Numpy数组添加噪声

##### 高斯噪声

![a+b](F:/Users/14024/Desktop/opencv_study/result/gasuss.png)

##### 椒盐噪声

![a+b](F:/Users/14024/Desktop/opencv_study/result/sp.png)

##### 随机噪声

![a+b](F:/Users/14024/Desktop/opencv_study/result/random.png)

#### 通过skimage库添加噪声

| 模块         | 功能                                     |
| ------------ | ---------------------------------------- |
| io           | 读取，保存和显示图像或视频               |
| data         | 提供测试图像和样本数据                   |
| color        | 颜色空间变换                             |
| filters      | 图像增强，边缘检测，排序滤波器，自动阈值 |
| draw         | 画图                                     |
| transform    | 几何变换                                 |
| morphology   | 形态学操作                               |
| exposure     | 图像强度调整                             |
| feature      | 特征提取                                 |
| measure      | 图像属性测量                             |
| segmentation | 图像分割                                 |
| restoration  | 图像恢复                                 |
| util         | 通用函数                                 |

![a+b](F:/Users/14024/Desktop/opencv_study/result/Figure_8.png)

### 直方图均衡化

#### 使用Matplotlib库绘制图像直方图

![a+b](F:/Users/14024/Desktop/opencv_study/result/Figure_9.png)

![a+b](F:/Users/14024/Desktop/opencv_study/result/Figure_10.png)

#### 使用OpenCV中的函数绘制直方图

`hist = cv2.calcHist(img, ch, mask, histSize, ranges [, accumulate])`

ch为通道的索引，[0]为直方图，[1],[2],[3]为BGR的通道

mask为图像掩码，完整的图像为None

histSize为BIN计数

ranges为范围，通常为[0,256]

![a+b](F:/Users/14024/Desktop/opencv_study/result/Figure_11.png)

#### 自定义函数实现直方图均衡化

代码有问题，不打算处理

#### 使用OpenCV函数实现直方图均衡化

`dst = cv2.equalizeHist(src)`

<img src="F:\Users\14024\Desktop\opencv_study\result\原始灰度直方图.png" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/均衡化直方图.png" alt="a+b" style="zoom: 50%;" />



<img src="F:/Users/14024/Desktop/opencv_study/result/Gray1.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/EqualizeHist.png" alt="a+b" style="zoom: 50%;" />

#### 自适应直方图均衡化

`dst = cv2.createCLAHE(clipLimit, titleGridSize)`

clipLimit:颜色对比度阈值

titleGridSize:均衡化的网格大小

<img src="F:/Users/14024/Desktop/opencv_study/result/原始直方图.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/自适应直方图均衡化.png" alt="a+b" style="zoom: 50%;" />

<img src="F:/Users/14024/Desktop/opencv_study/result/origin.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/CLAHE.png" alt="a+b" style="zoom: 50%;" />

### 直方图规定化

#### 自定义映像函数实现直方图规定化

效果不是很好

| 参考图 | 输入图 | 输出图 |
| ------ | ------ | ------ |



<img src="F:/Users/14024/Desktop/opencv_study/result/Reference Img.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Input Img.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Output Img.png" alt="a+b" style="zoom: 50%;" />

彩色图直方图规定化
| 参考图 | 输入图 | 输出图 |
| ------ | ------ | ------ |

<img src="F:/Users/14024/Desktop/opencv_study/result/Target.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Origin3.png" alt="a+b" style="zoom: 33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Specification Img.png" alt="a+b" style="zoom: 33%;" />

效果出奇的好

#### 直方图反向投影

`dst = cv2.normalize(src, dst, alpha, beta, norm, dtype)`

src：输入数组

dst：输出数组

alpha：范围下限

beta：范围上限

norm：范式-规定化类型。

​	NORM_MINMAX：线性归一化

​	NORM_INF：无穷范数

​	NORM_L1：1范数

​	NORM_L2：2范数

dtype：规定信道数和深度

`dst = cv2.calcBackProject(image, channels, hist, ranges, scale)`

channels：信道

hist：直方图

ranges：变化范围

scale：比例因子

效果成谜，问题出在值上

## 图像空域滤波

### 空域滤波

#### 线性空域滤波

$$
g(x,y)=\sum_{s=-a}^a\sum_{t=-b}^bw(s,t)f(x+s,y+t)
$$

其中
$$
a=\frac{m-1}{2},b=\frac{n-1}{2}
$$

#### 非线性空域滤波

1）忽略边界像素

2）保留原边界像素

### 图像平滑

#### 均值滤波

`dst = cv2.blur(src, ksize, anchor = None, brderType = None)`

ksize:表示滤波卷积核的大小

anchor:图像处理的锚点，默认为(-1,-1)

borderType:边界处理方式。

<img src="F:/Users/14024/Desktop/opencv_study/result/Origin4.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/N = 3.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/N = 7.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/N = 15.png" alt="a+b" style="zoom: 40%;" />

#### 方框滤波

`dst = cv2.boxFilter(src, ddopth, ksize, anchor, normalize, borderType)`

ddepth为处理结果图像的图像深度

ksize为滤波核的大小

normalize为是否归一化处理

<img src="F:/Users/14024/Desktop/opencv_study/result/Origin5.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/n = 0.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/n = 1.png" alt="a+b" style="zoom: 40%;" />

#### 高斯滤波

`dst = cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType = None)`

ksize为卷积核大小，必须是奇数

sigmaX为卷积核水平方向的权重

sigmaY为卷积和垂直方向的权重

borderType为边界处理方式

<img src="F:/Users/14024/Desktop/opencv_study/result/Origin5.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/N = 3 Gauss.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/N = 7 Gauss.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/N = 15 Gauss.png" alt="a+b" style="zoom: 40%;" />

#### 中值滤波

`dst = cv2.medianBlur(size, ksize)`

ksize为卷积核大小

<img src="F:/Users/14024/Desktop/opencv_study/result/Origin5.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/N = 3 median.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/N = 7 median.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/N = 15 median.png" alt="a+b" style="zoom: 40%;" />

#### 双边滤波

`dst = cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, borderType)`

d:滤波时选取的空间距离

sigmaColor:色差范围

sigmaSpace：滤波点数，值越大，滤波点越多

borderType：边界处理方式

<img src="F:/Users/14024/Desktop/opencv_study/result/Origin5.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/BF1.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/BF2.png" alt="a+b" style="zoom: 40%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/BF3.png" alt="a+b" style="zoom: 40%;" />

### 图像锐化

#### 拉普拉斯滤波

`dst = cv2.Laplacian(src, ddepth[,ksize[, scale[, delta[, borderType]]]])`

ddepth：图像深度：CV_8U,CV_16U,CV_16S,CV_32F,CV_64F

ksize：算子的大小

delta：可选的增量

<img src="F:/Users/14024/Desktop/opencv_study/result/Origin6.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Laplacian.png" alt="a+b" style="zoom: 50%;" />

#### 自定义卷积核滤波

`dst = cv2.filter2D(src, ddepth, kernel[, anchor[, delta[, borderType]]])`

kernel:卷积核

<img src="F:/Users/14024/Desktop/opencv_study/result/Origin1.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/K3.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/K5.png" alt="a+b" style="zoom: 50%;" />

#### 非锐化掩模和高频提升滤波

$$
g_{mask}(x,y)=f(x,y)-\overline{f}(x,y)
$$

$$
g(x,y)=f(x,y)+k*g_{mask}(x,y)
$$

<img src="F:/Users/14024/Desktop/opencv_study/result/Origin7.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Unsharp mask.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/High freq.png" alt="a+b" style="zoom: 50%;" />

####  练习见代码

## 图像频域滤波

### 傅里叶变换

#### Numpy中的傅里叶变换

$$
F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(ux/M+vy/N)}
$$

$$
f(x,y)=\sum_{u=0}^{M-1}\sum_{v=0}^{N-1}F(u,v)e^{-j2\pi(ux/M+vy/N)}
$$

`np.fft.fft2(src, n = None, axis = -1, norm = None)`

![a+b](F:/Users/14024/Desktop/opencv_study/result/傅里叶变换.png)

#### OpenCV中的傅里叶变换

`dst = cv2.dft(src, flags, nonzeroRows=0)`

flags:

|     标识符名称     |       意义       |
| :----------------: | :--------------: |
|    DFT_INVERSE     |  一维或二维变换  |
|     DFT_SCALE      |     缩放比例     |
|      DFT_ROWS      |     三维变换     |
| DFT_COMPLEX_OUTPUT | 一维或二维正变换 |
|  DFT_REAL_OUTPUT   | 一维或二维反变换 |

傅里叶逆变换

`iimg = cv2.idft(dft)`

求傅里叶逆变换后二维图像的幅值函数

`res2 = cv2.magnitude(x, y)`

<img src="F:/Users/14024/Desktop/opencv_study/result/Origin.png" alt="a+b" style="zoom:50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/FFT.png" alt="a+b" style="zoom:50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/IFFT.png" alt="a+b" style="zoom:50%;" />

### 低通滤波

#### 理想低通滤波

![a+b](F:/Users/14024/Desktop/opencv_study/result/Figure_12.png)

#### 巴特沃斯低通滤波

$$
H(u,v)=\frac{1}{1+(D(u,v)/D_0)^{2n}}
$$



![a+b](F:/Users/14024/Desktop/opencv_study/result/Img3.png)

#### 高斯低通滤波

$$
H(u,v)=e^{\frac{-D^2(u,v)}{2D_0^2}}
$$

![a+b](F:/Users/14024/Desktop/opencv_study/result/高斯低通滤波.png)

### 高通滤波

#### 理想高通滤波

![a+b](F:/Users/14024/Desktop/opencv_study/result/Ideal_high_pass.png)

#### 巴特沃斯高通滤波

$$
H(u,v)=1-\frac{1}{1+(D(u,v)/D_0)^{2n}}
$$

![a+b](F:/Users/14024/Desktop/opencv_study/result/buttworth_high.png)

#### 高斯高通滤波

$$
H(u,v)=1-e^{\frac{-D^2(u,v)}{2D_0^2}}
$$

![a+b](F:/Users/14024/Desktop/opencv_study/result/gas_high.png)

#### 带通和带阻滤波

#### 带通滤波

保留一范围的频率，其他信息滤掉

理想带通滤波器

巴特沃斯带通滤波器

高斯带通滤波器

![a+b](F:/Users/14024/Desktop/opencv_study/result/Ideal_bandpass.png)

![a+b](F:/Users/14024/Desktop/opencv_study/result/gaussian_bandpass.png)

![a+b](F:/Users/14024/Desktop/opencv_study/result/butterworth_bandpass.png)

#### 带阻滤波

把一范围的频率全部削掉，其他信息不滤波

理想带阻滤波器

巴特沃斯带阻滤波器

高斯带阻滤波器

![a+b](F:/Users/14024/Desktop/opencv_study/result/Ideal_bandstop.png)

![a+b](F:/Users/14024/Desktop/opencv_study/result/gaussian_bandstop.png)

![a+b](F:/Users/14024/Desktop/opencv_study/result/butterworth_bandstop.png)

### 同态滤波

将非线性的信号重新组合成线性信号

![a+b](F:/Users/14024/Desktop/opencv_study/result/homo_filtering.png)

## 图像退化和还原

图像退化是指图像因为某种原因变得不正常，有模糊，失真，有噪声

图像复原是指将图像建立退化模型，在进行反向推演，最终达到图像复原的目的

### 图像的运动模糊

$$
\int_{0}^{T} f[(x-x_0(t)),(y-y_0(t))] dx
$$

模型函数为motion_process(img_size, motion_angle)，包含两个参数：图像尺寸和运动角度

a<=45°时

`PSF[int(center_position+offset), int(center_postion-offset)]=1`

a>45°时

`PSF[int(center_position-offset), int(center_postion+offset)]=1`

![a+b](F:/Users/14024/Desktop/opencv_study/result/blur image.png)

### 图像的逆滤波

逆滤波是一种无约束的图像复原算法，其目标是找最优估计图像

![a+b](F:/Users/14024/Desktop/opencv_study/result/inverse.png)

### 图像的维纳滤波

是基于最小均方差准则提出的最佳线性滤波方法，滤波输出与期望输出的均方误差为最小，是一个最佳滤波系统

公式如下：
$$
F(u,v)=\frac{H(u,v)^*}{H^2(u,v)+k}G(u,v)
$$
![a+b](F:/Users/14024/Desktop/opencv_study/result/wiener.png)

### 图像质量的评价

均方误差：MSE

```python
def mse(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    return mse
```

峰值信噪比：PSNR
$$
PSNR = 10log_{10}\frac{MaxValue^2}{MSE}
$$

```python
def psnr(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 /mse)
```

结构相似性评价方法：SSIM
$$
SSIM(x,y)=\frac{2(\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma^2_x+\sigma_y^2+c_2)}
$$
`scipy.signal.convolve2d(in1, in2, mode = 'full', boundary = 'fill', fillvalue = 0)`

<img src="F:/Users/14024/Desktop/opencv_study/result/blur image.png" alt="a+b" style="zoom:50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Origin8.png" alt="a+b" style="zoom:50%;" />

MSE:134.89508406667613
PSNR:26.830842377461718
SSIM:134.89508406667613

### 练习见代码

## 图像数学形体学

### 结构元素

#### 使用OpenCV生成结果元素

`kernel=cv2.getStructuringElement(shape, ksize[, anchor])`

shape:内核形状

MORPH_RECT:产生矩阵的结构元素

MORPH_ELLIPSEM:产生椭圆的结构元素

MORPH_CROSS:产生十字交叉形的结构元素

ksize:内核的尺寸

anchor:内核锚点的位置

```matlab
[[1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]]
[[0 0 1 0 0]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [0 0 1 0 0]]
[[0 0 1 0 0]
 [0 0 1 0 0]
 [1 1 1 1 1]
 [0 0 1 0 0]
 [0 0 1 0 0]]
```

使用Numpy生成结构元素

```matlab
[[1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]]
[[0 0 1 0 0]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [0 0 1 0 0]]
[[0 0 1 0 0]
 [0 0 1 0 0]
 [1 1 1 1 1]
 [0 0 1 0 0]
 [0 0 1 0 0]]
[[1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]]
[[0 0 1 0 0]
 [0 1 1 1 0]
 [1 1 1 1 1]
 [0 1 1 1 0]
 [0 0 1 0 0]]
```

### 腐蚀

指卷积核沿着图像滑动，把物体的边界腐蚀掉。

`dst = cv2.erode(src, element[,anchor[, iterations[, borderType[, borderValue]]]])`

|  输入参数   |          意义           |
| :---------: | :---------------------: |
|     src     |      输入的原图像       |
|   element   |        结构元素         |
|   anchor    |     结构元素的锚点      |
| iterations  | 腐蚀操作的次数，默认为1 |
| borderType  |      边界扩充类型       |
| borderValue |       边界扩充值        |

![a+b](F:/Users/14024/Desktop/opencv_study/result/erode.png)

#### skimage中的腐蚀函数

`dst = skimage.morphology.erosion(image, selem = None)`

selem表示结构元素，用于设定局部区域的形状和大小

如果处理图像为二值图像，则函数为

`dst =  skimage.morphology.binary_erosion(image, selem = None)`

![a+b](F:/Users/14024/Desktop/opencv_study/result/morphology.png)

### 膨胀

#### OpenCV中的膨胀函数

`dst = cv2.dilate(src, element[,anchor[, iterations[, borderType[, borderValue]]]])`

![a+b](F:/Users/14024/Desktop/opencv_study/result/erode1.png)

#### skimage中的膨胀函数

`dst = skimage.morphology.dilation(image, selem = None)`

如果是二值图像

`dst = skimage.morphology.binary_dilation(image, selem = None) `

![a+b](F:/Users/14024/Desktop/opencv_study/result/morphology1.png)

卷积核的形状

|         函数         |   形状   |
| :------------------: | :------: |
|   morphlogy.square   |  正方形  |
|    morphlogy.disk    | 平面圆形 |
|    morphlogy.ball    |   球形   |
|    morphlogy.cube    | 立方体形 |
|  morphlogy.diamond   |  钻石形  |
| morphlogy.rectangle  |   矩形   |
|    morphlogy.star    |   星形   |
|  morphlogy.octagon   |  八角形  |
| morphlogy.octahedron |  八面体  |



#### OpenCV形态学处理原型函数

`dst = cv2.morphologyEx(src, op, kernel[,anchor[,iterations[,borderType[,borderValue]]]])`

op为形态学操作的类型，如下：

|          模式          |    描述    |
| :--------------------: | :--------: |
|    cv2.MORPH_ERODE     |    腐蚀    |
|    cv2.MORPH_DILATE    |    膨胀    |
|     cv2.MORPH_OPEN     |   开运算   |
|    cv2.MORPH_CLOSE     |   闭运算   |
|   cv2.MORPH_GRADIENT   |  形态梯度  |
|    cv2.MORPH_TOPHAT    |  高帽运算  |
|   cv2.MORPH_BLACKHAT   |  黑帽运算  |
| c2.MORPH_MORPH_HITMISS | 击中击不中 |

### 开运算

先腐蚀，在膨胀

### 闭运算

先膨胀，再腐蚀

#### opencv函数运算

![a+b](F:/Users/14024/Desktop/opencv_study/result/MORPH_OPEN_CLOSE.png)

#### 用skimage函数进行运算

`dst = skimage.morphology.openning(image, selem = None)`

`dst = skimage.morphology.closing(image, selem = None)`

二值图像的处理方式为

`dst = skimage.morphology.binary_openning(image, selem = None)`

`dst = skimage.morphology.binary_closing(image, selem = None)`

![a+b](F:/Users/14024/Desktop/opencv_study/result/Morphology_Open_Close.png)

### 高帽运算

将原图像减去他的开运算值，开运算可以消除暗背景下的较亮区域，可以得到原图像中灰度较亮的区域。高帽运算的一个作用就是校正不均匀光照，返回比结构元素小的白点

### 黑帽运算

将原图像减去他的闭运算值，可以删除亮度较高背景下的较暗区域，可以得到原图像中灰度较暗的区域。他能返回比结构化元素小的黑点，且将这些黑点反色

#### opencv函数运算

![a+b](F:/Users/14024/Desktop/opencv_study/result/MORPH_TOPHAT.png)

#### skimage函数运算

![a+b](F:/Users/14024/Desktop/opencv_study/result/morphology2.png)

### 形态学梯度

是根据膨胀结果减去腐蚀结果的差值，来实现增强结构元素领域中像素的强度，突出高亮区域的外围。形态学梯度的处理结果是图像中物体的边界，看起来像目标对象的轮廓

可以计算的梯度有如下四种：

1）**基本梯度：**膨胀图像减去腐蚀后的图像得到差值图像

2）**内部梯度：**用原图像减去腐蚀后的图像得到差值图像

3）**外部梯度：**膨胀图像剪掉原图像得到的差值图像

4）**方向梯度：**用X方向与Y方向的直线作为结构元素之后得到的图像梯度

<img src="F:/Users/14024/Desktop/opencv_study/result/original9.png" alt="a+b" style="zoom:33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/gradient.png" alt="a+b" style="zoom:33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/internal.png" alt="a+b" style="zoom:33%;" />

<img src="F:/Users/14024/Desktop/opencv_study/result/external.png" alt="a+b" style="zoom:33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/X-direction.png" alt="a+b" style="zoom:33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Y-direction.png" alt="a+b" style="zoom:33%;" />

创建滑动条，调整r结构元素大小和迭代次数i实现梯度运算

试试

```python
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
```

![a+b](F:/Users/14024/Desktop/opencv_study/result/morphology3.png)

效果还不错

### 灰度形态学

形态学运算应用到灰度图像中，用于提取描述和表示图像的某些特征，如图像边缘提取，平滑处理等

#### 灰度图像的腐蚀运算

腐蚀是取领域的最小值，从而减少高亮区域的面积

![a+b](F:/Users/14024/Desktop/opencv_study/result/edge.png)

边界提取的结果

#### 灰度图像的膨胀运算

膨胀是取领域的最大值，从而增大高亮区域的面积

![a+b](F:/Users/14024/Desktop/opencv_study/result/dilate1.png)

#### 灰度运算的开闭运算

<img src="F:/Users/14024/Desktop/opencv_study/result/morphology_open.png" alt="a+b" style="zoom: 33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/morphology_close.png" alt="a+b" style="zoom:33%;" />

### 形态学运算检测图像的边缘和角点

#### 检测图像边缘

<img src="F:/Users/14024/Desktop/opencv_study/result/Origin.png" alt="a+b" style="zoom: 33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/dilate.png" alt="a+b" style="zoom: 33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/erode2.png" alt="a+b" style="zoom: 33%;" />

<img src="F:/Users/14024/Desktop/opencv_study/result/absdiff.png" alt="a+b" style="zoom: 33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/threshold.png" alt="a+b" style="zoom: 33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/threshold.png" alt="a+b" style="zoom: 33%;" />

#### 检测图像角点



<img src="F:/Users/14024/Desktop/opencv_study/result/raw_img.png" alt="a+b" style="zoom: 33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/corners_0.05.png" alt="a+b" style="zoom: 33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/corners_0.01.png" alt="a+b" style="zoom: 33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/corners_0.005.png" alt="a+b" style="zoom: 33%;" />

### 击中与击不中运算

`dst = cv2.morphologyEx(src, cv2.MORPH_HITMISS, kernel)`

<img src="F:/Users/14024/Desktop/opencv_study/result/raw_img.png" alt="a+b" style="zoom: 33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/hit_miss.png" alt="a+b" style="zoom: 33%;" />

## 边缘检测

### Roberts算子

$$
G_x=f(i,j)-f(i-1,j-1)\\
G_y=f(i-1,j)-f(i,j-1)\\
|G(x,y)|=\sqrt{G^2_x+G^2_y}
$$

函数为

`dst = cv2.filter(src, ddepth, kernel[, anchor[, delta[, borderType]]])`

`dst = cv2.convertScaleAbs(src[,alpha[,beta]])`

`dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[,dtype])`

<img src="F:/Users/14024/Desktop/opencv_study/result/Roberts.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Roberts_SP.png" alt="a+b" style="zoom: 50%;" />

### Prewitt算子

是一阶微分算子的边缘检测，结合了差分运算与领域平均的方法。Prewitt算子在检测边缘时，去掉了部分伪边缘，对噪声有平滑作用

<img src="F:/Users/14024/Desktop/opencv_study/result/Prewitt.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Prewitt_sp.png" alt="a+b" style="zoom: 50%;" />

### Sobel算子

在prewitt算子的基础上增加了权重的概念，并结合高斯平滑和微积分求导。他认为相邻点的距离对当前像素点的影响是不同的，距离越近，影响越大。从而实现图像锐化并突出边缘轮廓

`dst = cv2.Sobel(src, ddepth, dx, dy[, ksize[,scale[,delta[,borderType]]]])`

dx和dy表示x或y方向求导的阶数

<img src="F:/Users/14024/Desktop/opencv_study/result/Sobel X.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Sobel Y.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Sobel Combined.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Sobel X_sp.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Sobel Y_sp.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Sobel Combined_sp.png" alt="a+b" style="zoom: 50%;" />

### 拉普拉斯算子

$$
\nabla^2f(x,y)=\frac{\delta^2f(x,y)}{\delta x^2}+\frac{\delta^2f(x,y)}{\delta y^2}
$$

在x方向上：
$$
\frac{\delta^2f}{\delta x^2}=f(x+1,y)+f(x-1,y)-2f(x,y)
$$
在y方向上：
$$
\frac{\delta^2f}{\delta y^2}=f(x,y+1)+f(x,y-1)-2f(x,y)
$$
近似为：
$$
\nabla^2f(x,y)=f(x+1,y)+f(x-1,y)+f(x,y-1)+f(x,y+1)-4f(x,y)
$$
`dst = cv2.Laplacian(src, ddepth, ksize[, scale[,delta[, borderType]]])`

<img src="F:/Users/14024/Desktop/opencv_study/result/Laplacian1.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Laplacian3.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Laplacian5.png" alt="a+b" style="zoom: 50%;" />

<img src="F:/Users/14024/Desktop/opencv_study/result/Laplacian1_sp.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Laplacian3_sp.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Laplacian5_sp.png" alt="a+b" style="zoom: 50%;" />

### Canny算子

$$
M(x,y)=\sqrt{g_x^2+g_y^2},\alpha(x,y)=arctan[\frac{g_{xx}}{g_y}]
$$

`dst = cv2.Canny(src, threshold1, threshold2[,apertureSize[, L2gradient]])`

threshold1表示低阈值

threshold2表示高阈值

apertureSize表示sobel算子大小

L2gradient表示是否用更精确的布尔值计算

<img src="F:/Users/14024/Desktop/opencv_study/result/canny.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Color dst.png" alt="a+b" style="zoom: 50%;" />

<img src="F:/Users/14024/Desktop/opencv_study/result/canny_sp.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Color dst_sp.png" alt="a+b" style="zoom: 50%;" />

### Scharr算子

Scharr算子是对Sobel算子差异性的增强

`dst = cv2.Scharr(src, ddepth, dx, dy[, scale[, delta[, borderType]]])`

<img src="F:/Users/14024/Desktop/opencv_study/result/Scharr X.png" alt="a+b" style="zoom: 50%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Scharr Y.png" alt="a+b" style="zoom: 50%;" /><img src="F:\Users\14024\Desktop\opencv_study\result\Scharr.png" style="zoom:50%;" />

### Kirsch算子和Robinson算子

Kirsch由八个方向的卷积核构成，代表8个方向，对图像的8个特定边缘方向做出最大相应，最大值作为边缘输出

ROBinson算子也是如此

### 高斯拉普拉斯算子

高斯函数的公式如下：
$$
G_\delta(x,y)=exp(-\frac{x^2+y^2}{2\delta^2})
$$
LoG算子的表达式如下：
$$
LoG =\nabla G_\delta(x,y)=\frac{\delta^2G_\delta(x,y)}{\delta x^2}+\frac{\delta^2G_\delta(x,y)}{\delta y^2}=\frac{x^2+y^2-2\delta^2}{\delta ^4}e^{-(x^2+y^2)/2\delta^2}
$$
<img src="F:/Users/14024/Desktop/opencv_study/result/LoG.png" alt="a+b" style="zoom: 50%;" />

### 高斯差分算子

$$
DoG = G_{\delta_1}-G_{\delta_2}=\frac{1}{2\pi}[\frac{1}{\delta_1^2}e^{-(x^2-y^2)/2\delta_1^2}-\frac{1}{\delta_2^2}e^{-(x^2-y^2)/2\delta_2^2}]
$$

### 霍夫变换

#### 霍夫变换检测直线

`lines = cv2.HoughLines(img, rho, theta, threshold)`

rho和theta分别是r，theta的精度

threshold是阈值

<img src="F:/Users/14024/Desktop/opencv_study/result/houghlines.png" alt="a+b" style="zoom: 200%;" />

在许多情况下会导致虚假检测，所以有概率霍夫变换

`lines = cv2.HoughLinesP(img, rho, theta, threshold[, minLineLength[, maxLineGap]])`

minLineLength是以像素为单位的最小长度

maxLineGap判定为一条线段的最大允许间隔

<img src="F:/Users/14024/Desktop/opencv_study/result/houghlinesp.png" alt="a+b" style="zoom: 200%;" />

#### 霍夫变换检测圆环

`circles = cv2.HoughCircles(img, method, dp, minDist, param1, param2, minRadius, maxRadius)`

method只有cv2.HOUGH_GRADIENT

dp越大，累加器数组越小

minDist为点到圆中心的最小距离

param1为处理边缘检测的梯度值方法

param2为累加器阈值

minRadius为最小半径

```python
import cv2
import numpy as np

img = cv2.imread("picture_material/coin.jpg")
# img = cv2.resize(img, (int(img.shape[1]*5), int(img.shape[0]*5)), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("origin", gray)

#hough变换
circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 
100, param1 = 100, param2 = 30, minRadius = 180, maxRadius = 185)
circles = circles1[0, :, :]
circles = np.uint16(np.around(circles))
for i in circles[:]:
    cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 5)#画圆
    cv2.circle(img, (i[0], i[1]), 2, (255, 0, 255), 10)#画圆心
    
cv2.imshow("result",img)
cv2.waitKey()
```

效果不是很好

![a+b](F:/Users/14024/Desktop/opencv_study/result/coin.png)

## 图像分割

### 图像阈值分割

$$
g(x,y)=
\left\{  
             \begin{array}{**lr**}  
             1 & f(x,y) \geq T\\  
            0 & f(x,y) \leq T\\      
             \end{array}  
\right.
$$

#### 全局阈值分割

当图像高于某个阈值时赋予白色，低于某个阈值赋予黑色

`dst = cv2.threshold(src, thresh, maxval, type)`

|  函数  |        意义        |
| :----: | :----------------: |
| thresh |   阈值 取值0~255   |
| maxval |  填充色 取值0~255  |
|  type  | 阈值类型，类型如下 |

|         类型          |           含义            |
| :-------------------: | :-----------------------: |
|   cv2.THRESH_BINARY   |          二值化           |
| cv2.THRESH_BINARY_INV |         反二值化          |
|   cv2.THRESH_TRUNC    | 截断阈值化，大于阈值设为1 |
|   cv2.THRESH_TOZERO   | 阈值化为0，小于阈值设为0  |
| cv2.THRESH_TOZERO_INV |       大于阈值设为0       |

![a+b](F:/Users/14024/Desktop/opencv_study/result/Figure_13.png)

#### 自适应阈值

自适应阈值二值化函数根据图像的小块区域的值来计算对应区域的阈值，从而得到更为适合的图像

`dst = cv2.adaptiveThreshold(src, maxval, thresh_type, type, Block_Size, C)`

|    参数     |                             含义                             |
| :---------: | :----------------------------------------------------------: |
| thresh_type | 计算阈值的方法有如下两种：cv2.ADAPTIVE_THRESH_MEAN_C(平均法)\cv2.ADAPTIVE_THRESH_GAUSSIAN_C(高斯法) |
|    type     |                           阈值类型                           |
| Block_Size  |                          分块的大小                          |
|      C      |                            常数项                            |

![a+b](F:/Users/14024/Desktop/opencv_study/result/adaptiveThreshold.png)

#### Otsu's二值化

1）计算图像直方图

2）设置阈值，大于阈值的一组，小于阈值的一组

3）分别计算两组的偏移数

4）把0~255依照顺序设为阈值，重复上述步骤，直到得到最小偏移数

![a+b](F:/Users/14024/Desktop/opencv_study/result/otsus.png)

### 图像区域分割

主要方法有区域生长和区域分裂合并法，都是典型的串行区域技术，其后续的处理根据前面的步骤决定

#### 区域生长

![a+b](F:/Users/14024/Desktop/opencv_study/result/SeedMark.png)

#### 区域分裂合并

从整个图像出发，不断分裂到各个子区域，然后把前景区域合并，实现目标提取

![a+b](F:/Users/14024/Desktop/opencv_study/result/spilitting.png)

### 图像的边缘分割

通过边缘检测，检测灰度级或结构有突变的地方，表明一个区域的终结，也是一个区域的开始

<img src="F:/Users/14024/Desktop/opencv_study/result/binary.png" alt="a+b" style="zoom: 33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/inv.png" alt="a+b" style="zoom: 33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/canny1.png" alt="a+b" style="zoom: 33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/inv_canny.png" alt="a+b" style="zoom: 33%;" />

### 直方图分割法

直方图分割法有明显的双峰

![a+b](F:/Users/14024/Desktop/opencv_study/result/hist_edge.png)

### 图像连接组件标记算法

**连通组件标记函数**

`retval, labels = cv2.connctedComponents(img, connectivity, ltype)`

img:二值图像

connectivity:8连通域

ltype:输出的labels类型，默认是CV_32S

retval,labels:输出的标记图像

**连通组件状态统计函数**

`retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, ltype)`

其中，在stats中

|      组件      |                 含义                  |
| :------------: | :-----------------------------------: |
|  CC_STAT_LEFT  | 连通组件外接矩形左上角坐标的X位置信息 |
|  CC_STAT_TOP   | 连通组件外接矩形左上角坐标的Y位置信息 |
| CC_STAT_WIDTH  |         连通组件外接矩形宽度          |
| CC_STAT_HEIGHT |         连通组件外接矩形高度          |
|  CC_STAT_AREA  |          连通组件的面积大小           |

![a+b](F:/Users/14024/Desktop/opencv_study/result/colors labels.png)

## 彩色图像的处理

### 彩色模型

RGB模型（通用模型）

CMY模型（青，品红，黄）和CMYK模型（青，品红，黄，黑）用于彩色打印机

HSI模型（色调，亮度，饱和度）人类描述的颜色

CIE模型

#### RGB彩色模型

RGB彩色模型基于笛卡尔坐标系，8个顶点颜色如下

| 彩色名称 |  R   |  G   |  B   |
| :------: | :--: | :--: | :--: |
|    黑    |  0   |  0   |  0   |
|    蓝    |  0   |  0   |  1   |
|    绿    |  0   |  1   |  0   |
|   蓝绿   |  0   |  1   |  1   |
|    红    |  1   |  0   |  0   |
|   品红   |  1   |  0   |  1   |
|    黄    |  1   |  1   |  0   |
|    白    |  1   |  1   |  1   |

#### CMY和CMYK模型

CMY转换为CMYK的公式为

```
K = min(C,M,Y)
C = C - K
M = M - K
Y = Y - K
```



#### HSI彩色模型

RGB转换为HSI模型公式
$$
I=\frac{1}{3}(R+G+B)\\
S = 1-\frac{1}{(R+G+B)}[min(R,G,B)]\\
H=\left\{  
             \begin{array}{**lr**}  
             \theta & G \geq B\\  
            2\pi - \theta & G<B\\      
             \end{array}  
\right.\\
其中：\\
\theta=arccos(\frac{\frac{1}{2}[(R-G)+(R-B)]}{[(R-G)^2+(R-B)(G-B)^{\frac{1}{2}}]})
$$
HSI转换为RGB模型公式

当H在[0,2Π/3]时
$$
B=I(1-S)\\
R=I[1+\frac{ScosH}{cos(\frac{\pi}{3}-H)}]\\
G=3I-(B+R)
$$
当H在[2Π/3,4Π/3]时
$$
R=I(1-S)\\R=I[1+\frac{Scos(H-\frac{2\pi}{3})}{cos(\pi-H)}]\\G=3I-(B+R)
$$
当H在[4Π/3,2Π]时
$$
G=I(1-S)\\
R=I[1+\frac{Scos(H-\frac{4\pi}{3})}{cos(\frac{5\pi}{3}-H)}]\\
R=3I-(B+G)
$$

#### YIQ彩色模型

RGB到YIQ转换公式为
$$
\left[
\begin{matrix}
Y\\
I\\
Q
\end{matrix}
\right]
=
\left[
\begin{matrix}
0.299 & 0.587 & 0.114\\
0.596 & -0.274 & -0.322\\
0.211 & -0.253 & 0.312
\end{matrix}
\right]
\left[
\begin{matrix}
R\\
G\\
B
\end{matrix}
\right]
$$

#### YCrCb彩色模型

$$
\left[
\begin{matrix}
Y\\
Cb\\
Cr
\end{matrix}
\right]
=
\left[
\begin{matrix}
0.299 & 0.587 & 0.114\\
-0.169 & -0.331 & -0.500\\
0.500 & -0.419 & -0.312
\end{matrix}
\right]
\left[
\begin{matrix}
R\\
G\\
B
\end{matrix}
\right]
$$

### 色彩空间类型转换

`img = cv2.cvtColor(src, code, dstCn)`

#### RGB色彩空间

#### GRAY色彩空间

这个我就不写了啊，做的太多了

#### YCrCb色彩空间

![a+b](F:/Users/14024/Desktop/opencv_study/result/YCrCb.png)

#### HSV色彩空间

$$
S=
\left\{  
             \begin{array}{**lr**}  
             \frac{V-min(R,G,B)}{V} & V不等于0\\  
            0 & 其他\\      
             \end{array}  
\right.\\
H=
\left\{  
             \begin{array}{**lr**}  
             \frac{60(G-B)}{V-min(R,G,B)} & V=R\\  
            120+\frac{60(B-R)}{V-min(R,G,B)} & V=G\\    
            240+\frac{60(R-G)}{V-min(R,G,B)} & V=G\\  
             \end{array}  
\right.\\
其中：\\
V=max(R,G,B)\\
H<0时\\
H=
\left\{  
             \begin{array}{**lr**}  
             H+360 & H<0\\  
            H & 其他\\     
             \end{array}  
\right.\\
$$

![a+b](F:/Users/14024/Desktop/opencv_study/result/HSV.png)

### 彩色图像通道的分离与合并

#### 彩色图像通道的分离

`b,g,r=cv2.split(src)`

<img src="F:/Users/14024/Desktop/opencv_study/result/Origin6.png" alt="a+b" style="zoom:33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Red.png" alt="a+b" style="zoom:33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Green.png" alt="a+b" style="zoom:33%;" /><img src="F:/Users/14024/Desktop/opencv_study/result/Blue.png" alt="a+b" style="zoom:33%;" />

带颜色的RGB三通道图像

<img src="F:\Users\14024\Desktop\opencv_study\result\R_img.png" style="zoom:33%;" /><img src="F:\Users\14024\Desktop\opencv_study\result\G_img.png" style="zoom:33%;" /><img src="F:\Users\14024\Desktop\opencv_study\result\B_img.png" style="zoom:33%;" />

#### 彩色图像通道的合并

`merged = cv2.merge([b, g, r])`

### 全彩色图像处理

#### 直方图处理

`img = cv2.equalizeHist(src)`

`matplotlib.pyplot.hist(img,BINS)`

#### 彩色图像的平滑和锐化

#### 基于彩色的图像分割

这两个以前做过，这里不赘述

## 图像特征的提取与描述

### 图像轮廓特征

#### 图像轮廓的查找和绘制

`contours, hierarchy = cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]])`

image为输入的图像，mode为轮廓检索模式

|     模式      |             说明             |
| :-----------: | :--------------------------: |
| RETR_EXTERNAL |      只检索最外面的轮廓      |
|   RETR_LIST   |  检索所有轮廓，保存在list中  |
|  RETR_CCOMP   |  检索所有的轮廓，并分为两层  |
|   RETR_TREE   | 检查所有轮廓，并重构整个层次 |

第二个参数是方法：

|            参数            |            意义             |
| :------------------------: | :-------------------------: |
|   cv2.CHAIN_APPROX_NONE    |        显示所有轮廓         |
|  cv2.CHAIN_APPROX_SIMPLE   |         只保留端点          |
|  cv2.CHAIN_APPROX_TX89_L1  |    使用teh-Chinl近似算法    |
| cv2.CHAIN_APPROX_TC89_KCOS | 使用teh-Chinl chain近似算法 |

轮廓绘制函数

`cv2.drawCountours(img, countours, -1, (0, 0, 255), 2)`

![a+b](F:/Users/14024/Desktop/opencv_study/result/Contour img.png)

#### 带噪声的轮廓

#### 边缘检测后的轮廓

![a+b](F:/Users/14024/Desktop/opencv_study/result/drawcontours_edge.png)

### 图像的几何特征

面积

`area = cv2.contoursArea(cnt, True)`

True表示是闭合轮廓还是曲线

周长

`perimeter = cv2.arcLength(cnt, True)`

输出

Area =  368095.0 Length =  2528.0

#### 外接矩形

##### 直角矩形

最上面，最下面，最左边和最右边的点的坐标

`x, y, w, h = cv2.boundingRect(cnt)`

左上角的坐标和矩形的宽和高

`img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)`

##### 旋转矩形

`min_rect = cv2.minAreaRect(cnt)`

实现方法：

```python
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
```

![a+b](F:/Users/14024/Desktop/opencv_study/result/Rectangles.png)

### 最小外接圆和椭圆

#### 最小外接圆

获得圆的半径和坐标

`(x, y), radius = cv2.minEnclosingCircle(cnt)`

画圆

`cv2.cirlce(img, center, radius, color[, thickness[, lineType[, shift]]])`

thickness:圆的厚度

lineType:圆边界的类型

shift:中心坐标和半径值的小数位数

#### 内接椭圆

`ellipse = cv2.fitEllipse(cnt)`

ellipse是长短轴的长度

画椭圆

cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]])

axes:椭圆尺寸

angle:旋转角度

startAngle:起始角度

endAngle:终止角度

![a+b](F:/Users/14024/Desktop/opencv_study/result/draw circle.png)

#### 近似轮廓

`cv2.aprroxPolyDP(cnt, epsilon, True)`

![a+b](F:/Users/14024/Desktop/opencv_study/result/approxPloyDP 0.1%.png)

#### 轮廓凸包

`hull = cv2.convexHull(points, clockwise, returnpoints)`

clockwise表示顺时针或是逆时针

returnpoints为False表示返回缺陷点

![a+b](F:/Users/14024/Desktop/opencv_study/result/line & points.png)

#### 拟合直线

`[vx, vy, x, y] = cv2.fitLine(points, distType, param, reps, aeps)`

points:待拟合的直线合集

distType:距离类型

|     函数      |                      类型                      |
| :-----------: | :--------------------------------------------: |
| cv2.DIST_USER |                   自定义距离                   |
|  cv2.DIST_L1  |                     1范数                      |
|  cv2.DIST_L2  |                     2范数                      |
|  cv2.DIST_C   |                    无穷范数                    |
| cv2.DIST_L12  |      distance=2(sqrt(1 + x * x / 2)  - 1)      |
| cv2.DIST_FAIR | distance=c^2(\|x\|/c-log(1+\|x\|/c))  c=1.3998 |

![a+b](F:/Users/14024/Desktop/opencv_study/result/Line.png)

### 图像特征矩

特征矩包括信息有：大小，位置，角度， 形状等被广泛的用在模式识别和图像分类方面

获得矩特征：

`retval = cv2.moments(array[, binaryImage])`

#### Hu矩

Hu矩是归一化中心距的线性组合，是进行旋转，缩放，平移后仍能保持矩的不变性

`hu = cv2.HuMoments(m)`

#### 形状匹配

'dist = cv2.matchShapes(contour1, contour2, method, parameter)'

可用于数字识别

![a+b](F:/Users/14024/Desktop/opencv_study/result/matchshape.png)

效果不好

### 图像匹配

#### ORB特征检测+暴力检测

`orb = cv2.ORB_create()`

暴力匹配

`bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCherk = True)`

![a+b](F:/Users/14024/Desktop/opencv_study/result/matches.png)

### 搜索匹配的图像

# 综合应用实例

## 1.车辆识别

![a+b](F:/Users/14024/Desktop/opencv_study/result/video.png)

## 2.人脸识别

### Haar人脸识别

创建Haar级联器

导入图片并将其灰度化

调用detectMultiScale方法进行人脸识别

`detectMultiScale(img, double scaleFactor = 1.1, int Neighbors = 3)`

![a+b](F:/Users/14024/Desktop/opencv_study/result/face.png)

#### 眼部识别

![a+b](F:/Users/14024/Desktop/opencv_study/result/eye.png)

## 3.车牌识别

车牌二值化处理

形态学处理

低通滤波去噪点 

缩放

![a+b](F:/Users/14024/Desktop/opencv_study/result/plate.png)![a+b](F:/Users/14024/Desktop/opencv_study/result/roi_bin.png)

# 深度学习基础

## 1.学习网络模型

DNN（深度神经网络）

RNN（循环神经网络）

CNN（卷积神经网络）

### RNN

语音识别

机器翻译

生成图像描述

### CNN

图像分类检索

目标定位检测

目标分割

人脸识别

## 2.OPENCV支持的模型

TensorFlow

Pytorch/torch

Caffe

DarkNet

## 3.DNN使用步骤

1.读取模型，得到深度学习网络

2.读取图片/视频

3.将图片转换成张量，送入深度神经网络

4.分析并得到结果

**导入模型**

```python
readNetFromTensorflow(model, config)#模型+参数
readNetFromTensorCaffe(config, model)	
readNetDarknet,YOLO
readnet(model, [config,[framework]])#framework：框架（可不写）
```

**转换张量**

```python
blobFromImage(img, scalefactor = 1.0, size = Size(), mean = Scalar(), swapRB = false, crop = false)
```

mean:减去光照均值

**将张量送入网络并执行**

```
net.setInput(blob)
net.forward()
```



