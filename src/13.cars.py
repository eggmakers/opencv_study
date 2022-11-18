import cv2
import numpy as np

cap = cv2.VideoCapture('picture_material/cars.mp4')

bgsubmog = cv2.bgsegm.createBackgroundSubtractorMOG()

#卷积核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

min_w = 27
min_h = 27
max_w = 140
max_h = 140

#检测线的高度
line_high = 700

cars = []
car_num = 0

#偏移量
offset = 1

def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

while True:
    ret, frame = cap.read()
    
    if ret == True:
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print(frame.shape)
        # exit()
        blur = cv2.GaussianBlur(frame, (3, 3), 5)
        #去背影
        mask = bgsubmog.apply(blur)
        
        #腐蚀
        erode = cv2.erode(mask, kernel, iterations = 1)
        
        #膨胀
        dilate = cv2.dilate(mask, kernel, kernel, iterations = 2)
        
        #闭操作
        close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
        close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
        # close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
        
        cnts, h = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.line(frame, (10, line_high), (1200, line_high), (255, 255, 0), 1)
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            isValid = w >= min_w and h >= min_h and w <= max_w and h <= max_h
            if(not isValid):
                continue
            
            #计算有效的车辆数目
            cpoint = center(x, y, w, h)
            cars.append(cpoint)
            # print(y)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            for(x, y) in cars:
                if((y > line_high - offset) and (y < line_high + offset)):
                    car_num +=1
                    cars.remove((x, y))
                    print("car number = ", car_num)
        
        cv2.putText(frame, "Car Count:" + str(car_num), (500, 60), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 0), 5)
        cv2.imshow("video", frame)
        # cv2.imshow("erode", close)
        
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()