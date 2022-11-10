import cv2
import numpy as np

def gradient_basic(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("gradient", dst)
    
def gradient_in_out(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dm = cv2.dilate(img, kernel)
    em = cv2.erode(img, kernel)
    dst1 = cv2.subtract(img, em)
    dst2 = cv2.subtract(dm, img)
    cv2.imshow("internal", dst1)
    cv2.imshow("external", dst2)
    
def gradient_X(img):
    kernel_x = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype = np.uint8)
    dmx = cv2.dilate(img, kernel_x)
    emx = cv2.erode(img, kernel_x)
    dstx = cv2.subtract(dmx, emx)
    cv2.imshow("X-direction", dstx)
    
def gradient_Y(img):
    kernel_y = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype = np.uint8)
    dmy = cv2.dilate(img, kernel_y)
    emy = cv2.erode(img, kernel_y)
    dsty = cv2.subtract(dmy, emy)
    cv2.imshow("Y-direction", dsty)
    
src = cv2.imread("picture_material/beauty_leg2.jpg")
src = cv2.resize(src, (int(src.shape[1]/2), int(src.shape[0]/2)), interpolation=cv2.INTER_AREA)
cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
cv2.imshow("original", src)
gradient_basic(src)
gradient_in_out(src)
gradient_X(src)
gradient_Y(src)
cv2.waitKey()