import cv2
import numpy as np
import matplotlib.pyplot as plt

def log_plot(c):
    x = np.arange(0, 255, 0.01)
    y = c * np.log(1 + x)
    plt.plot(x, y, 'r', linewidth=1)
    plt.title('logarithmic')
    plt.xlim(0, 255), plt.ylim(0, 255)
    plt.show()
    
def log(c, img_Gray):
    output = c * np.log(1.0 + img_Gray)
    output = np.uint8(output + 0.5)
    return output

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, ((int(img.shape[1]/2), int(img.shape[0]/2))), interpolation=cv2.INTER_AREA)
img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
c = 45
log_plot(c)
result = log(c, img_Gray)

cv2.imshow("Origin", img_Gray)
cv2.imshow("LOG CHG", result)
cv2.waitKey()
cv2.destroyAllWindows()