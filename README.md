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

```
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
