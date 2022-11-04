import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_plot(c, gamma):
    x = np.arange(0, 255, 0.01)
    #y = c * x ** gamma
    y = np.power(x, gamma)
    plt.plot(x, y, 'b', linewidth=1)
    plt.xlim([0, 255]), plt.ylim([0, 255])
    
def gamma_trans(img, c, gamma1):
    output_img = c * np.power(img / float(np.max(img)), gamma1) * 255.0
    output_img = np.uint8(output_img)
    return output_img

img = cv2.imread("picture_material/beauty_leg3.jpg")
img = cv2.resize(img, ((int(img.shape[1]/2), int(img.shape[0]/2))), interpolation=cv2.INTER_AREA)
img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(1), gamma_plot(1, 0.5),plt.title('gamma = 0.5')#伽马变换曲线
plt.figure(2), gamma_plot(1, 2),plt.title('gamma = 2')
plt.show()
result = gamma_trans(img_Gray, 1, 0.5)                    #图像伽马变换
result1 = gamma_trans(img_Gray, 1, 2.0)

cv2.imshow("Origin", img_Gray)
cv2.imshow("Gamma < 1", result)
cv2.imshow("Gamma > 1", result1)
cv2.waitKey()
cv2.destroyAllWindows()