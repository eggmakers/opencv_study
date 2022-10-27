import cv2
path = 'f:/Users/14024/Desktop/python_opencv学习/picture_material/WeChat_20220125175400.mp4'
videoCapture = cv2.VideoCapture(path)

fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fnums = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

video_writer = cv2.VideoWriter('f:/Users/14024/Desktop/python_opencv学习/picture_material/test.avi',cv2.VideoWriter_fourcc('X','V','I','D'),fps,size)

success,frame = videoCapture.read()
while success:
    cv2.imshow('Windows',frame) #显示
    cv2.waitKey((int(100/fps)))
    video_writer.write(frame)  #写入下一帧
    success,frame = videoCapture.read() #获取下一帧
    
videoCapture.release()
cv2.destroyAllWindows()