import cv2
img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
cv2.imshow("origin", img)

def nothing(*args):
    pass

#显示膨胀效果的窗口
cv2.namedWindow('dilate', 6)
r, MAX_R = 1, 20
#调整结构原半径
cv2.createTrackbar('r', 'dilate', r, MAX_R, nothing)

while True:
    r = cv2.getTrackbarPos('r', 'dilate')#得到当前r值
    s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    d = cv2.dilate(img, s)
    cv2.imshow('dilate', d)
    
    ch = cv2.waitKey(1)
    if ch == 27:
        break
cv2.destroyAllWindows()
