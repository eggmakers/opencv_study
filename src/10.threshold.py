import cv2
from matplotlib import pyplot as plt

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

title = ['Origin image', 'Binary', 'Binary_INV', 'Trunc', 'Tozero', 'Tozero_inv']
image = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3 ,i + 1)
    plt.imshow(image[i], 'gray')
    plt.title(title[i])
    plt.axis('off')
plt.show()