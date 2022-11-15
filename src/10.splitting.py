import numpy as np
import cv2

def judge(w0, h0, w, h):
    a = img[h0 : h0 + h, w0 : w0 + w]
    ave = np.mean(a)
    std = np.std(a, ddof = 1)
    count = 0
    total = 0
    for i in range(w0, w0 + w):
        for j in range(h0, h0 + h):
            if abs(img[j, i] - ave) < 1 * std:
                count += 1
            total += 1
    if (count / total) < 0.95:
        return True
    else:
        return False
    
#图像二值化处理
def draw(w0, h0, w, h):
    for i in range(w0, w0 + w):
        for j in range(h0, h0 + h):
            if img[j, i] > 125:
                img[j, i] = 255
            else:
                img[j, i] = 0
                
def splitting(w0, h0, w, h):
    if judge(w0, h0, w, h) and (min(w, h) > 5):
        splitting(w0, h0, int(w / 2), int(h / 2))
        splitting(w0 + int(w / 2), h0, int(w / 2), int(h / 2))
        splitting(w0, h0 + int(w / 2), int(w / 2), int(h / 2))
        splitting(w0 + int(w / 2), h0 + int(w / 2), int(w / 2), int(h / 2))
    else:
        draw(w0, h0, w, h)
        
if __name__ == '__main__':
    img = cv2.imread("picture_material/beauty_leg3.jpg")
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_input = img
    height, width = img.shape[0: 2]
    splitting(0, 0, width, height)
    cv2.imshow('input', img_input)
    cv2.imshow('output', img)
    cv2.waitKey()