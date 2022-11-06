import numpy as np
import cv2
img = cv2.imread("picture_material/beauty_leg3.jpg")
img = cv2.resize(img, (int(img.shape[1]/4),int(img.shape[0]/4)), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(image.shape)
def frequency_filter(image ,filter):
    """
    :param image:
    :param filter: 频域变换函数
    :return:
    """
    fftImg = np.fft.fft2(image) #对图像进行傅里叶变换
    fftImgShift = np.fft.fftshift(fftImg)#傅里叶变换后坐标移动到图像中心
    handle_fftImgShift1 = fftImgShift*filter#对傅里叶变换后的图像进行频域变换

    handle_fftImgShift2 = np.fft.ifftshift(handle_fftImgShift1)
    handle_fftImgShift3 = np.fft.ifft2(handle_fftImgShift2)
    handle_fftImgShift4 = np.real(handle_fftImgShift3)#傅里叶反变换后取频域
    return np.uint8(handle_fftImgShift4)

def ILPF(image,d0,n):#理想低通滤波器
    H = np.empty_like(image,dtype=float)
    M,N = image.shape
    mid_x = int(M/2)
    mid_y = int(N/2)
    for y in range(0, M):
        for x in range(0,N):
            d = np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2)
            if d <= d0:
                H[y, x] = 1**n
            else:
                H[y, x] = 0**n
    return H

def IHPF(img, d0, n):#理想高通滤波器
    H = np.empty_like(img, dtype = float)
    M, N = img.shape
    mid_x = int(M / 2)
    mid_y = int(N / 2)
    for y in range(0, M):
        for x in range(0, N):
            d = np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2)
            if d <= d0:
                H[y, x] = 0**n
            else:
                H[y, x] = 1**n
    return H

def BHPF(image,d0,n):#巴特沃斯高通滤波器
    H = np.empty_like(image,float)
    M,N = image.shape
    mid_x = int(M/2)
    mid_y = int(N/2)
    for y in range(0, M):
        for x in range(0, N):
            d = np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2)
            H[y,x] = 1-(1/(1+(d/d0)**(n)))
    return H

def GHPF(image,d0,n):#高斯高通滤波器
    H = np.empty_like(image,float)
    M, N = image.shape
    mid_x = M/2
    mid_y = N/2
    for x in range(0, M):
        for y in range(0, N):
            d = np.sqrt((x - mid_x)**2 + (y - mid_y) ** 2)
            H[x, y] = 1 - (np.exp(-d**n/(2*d0**n)))
    return H


def image_arrage(image,W,H,n,d0,step,filter):#图像绘制
    """
    :param image: 原始图像
    :param W: 每列图像个数
    :param H: 每行图像个数
    :param n: 阶数
    :param d0: 初始截止频率
    :param step: 截止频率步距
    :return: None
    """
    imageHstack = {}
    for i in range(H):
        hStack = 'H'+str(i)
        flag = 0
        for i in range(W):
            if flag ==0:
                imageHstack[hStack] = frequency_filter(image,filter(image,d0,n))
                d0 += step
                flag +=1
            else:
                imageHstack[hStack] = np.hstack((imageHstack[hStack], frequency_filter(image, filter(image, d0, n))))
                d0 += step
                flag += 1
    flag = 0
    for i in imageHstack.values():

        if  flag == 0:
            imageStack = i
            flag += 1
        else:
            imageStack = np.vstack((imageStack,i))
    # print(imageStack)
    return imageStack



# cv2.namedWindow('Img')
# cv2.resizeWindow('Img',(20,20))
# cv2.imshow('Img',frequency_filter(image,ILPF(image,60)))
# cv2.namedWindow('Img2')
# cv2.resizeWindow('Img2',(20,20))
# cv2.imshow('Img2',frequency_filter(image,BLPF(image,40,n=2)))
# cv2.namedWindow('Img3')
# cv2.resizeWindow('Img3',(20,20))
# imghstack = np.hstack((imgroi, imgwomen))
# # 垂直组合
# imgvstack = np.vstack((imgroi, imgwomen))
cv2.imshow('Img3',image_arrage(img,4,2,2,30,20,GHPF))

# cv2.resizeWindow('Img3',(20,20))
# cv2.imshow('Img3',frequency_filter(image,GLPF(image,80,n=2)))
cv2.waitKey()
