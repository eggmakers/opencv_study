import argparse
import copy
import cv2

parser = argparse.ArgumentParser()      #用于将命令行字符串解析为Python的对象。
parser.add_argument("--width", help="capture width", type=int, default=960)
parser.add_argument("--height", help="capture height", type=int, default=540)
args = parser.parse_args()

cap_width = args.width
cap_height = args.height
cap = cv2.VideoCapture("/picture_material/clock.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

while True:
    ret,frame = cap.read()
    if not ret:
        print('cap.read() error')
        break
    rotate_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    lin_polar_image = cv2.warpPolar(rotate_frame, (150, 500), (270, 480), 220, cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR)
    lin_polar_crop_image = copy.deepcopy(lin_polar_image[0:500, 15:135])
    lin_polar_crop_image = lin_polar_crop_image.transpose(1, 0, 2)[::-1]
    cv2.imshow('Original',frame)
    cv2.imshow('POLAR',lin_polar_crop_image)
    
    key = cv2.waitKey(50)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()