import cv2
import numpy as np

def originalSeed(gray, th):
    ret, thresh = cv2.threshold(gray, th, 250, cv2.THRESH_BINARY)
    #种子区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh_copy = thresh.copy()
    thresh_B = np.zeros(gray.shape, np.uint8)
    
    seeds = [ ]#记录种子坐标
    #清零像素值
    while thresh_copy.any():
        Xa_copy, Ya_copy = np.where(thresh_copy > 0)
        #thresh_A_copy中值为255的像素的坐标
        thresh_B[Xa_copy[0], Ya_copy[0]] = 255
        for i in range(200):
            dilation_B = cv2.dilate(thresh_B, kernel, iterations = 1)
            thresh_B = cv2.bitwise_and(thresh, dilation_B)
            
        #取thresh_B值为255的像素坐标，并将thresh_copy中对应的坐标像素值变为0
        Xb, Yb = np.where(thresh_B > 0)  
        thresh_copy[Xb, Yb] = 0
        
        #循环，导致有一个像素点为止
        while str(thresh_B.tolist()).count("255") > 1:
            thresh_B = cv2.erode(thresh_B, kernel, iterations = 1)#腐蚀操作
            
        X_seed, Y_seed = np.where(thresh_B > 0)
        if X_seed.size > 0 and Y_seed.size > 0:
            seeds.append((X_seed[0], Y_seed[0]))
        thresh_B[Xb, Yb] = 0
    return seeds

#区域生长
def regionGrow(gray, seeds, thresh, p):
    seedMark = np.zeros(gray.shape)
    if p == 8:
        connection = [(-1, -1), (-1, 0), (-1, -1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    elif p == 4:
        connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
    #seeds内无元素时生长停止
    while len(seeds) != 0:
        #栈顶元素出栈
        pt = seeds.pop(0)
        for i in range(p):
            tmpX = pt[0] + connection[i][0]
            tmpY = pt[1] + connection[i][1]
            
            #检测边界点
            if tmpX < 0 or tmpY < 0 or tmpX >= gray.shape[0] or tmpY >= gray.shape[1]:
                continue
            if abs(int(gray[tmpX, tmpY]) - int(gray[pt])) < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = 255
                seeds.append((tmpX, tmpY))
    return seedMark

if __name__ == "__main__":
    img = cv2.imread("picture_material/beauty_leg2.jpg")
    img = cv2.resize(img, ((int(img.shape[1]/2), int(img.shape[0]/2))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Origin", gray)
    seeds = originalSeed(gray, th = 180)
    seedMark = regionGrow(gray, seeds, thresh = 3, p = 8)
    cv2.imshow("SeedMark", seedMark)
    cv2.waitKey()