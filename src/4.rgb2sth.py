import cv2
from matplotlib import pyplot as plt

img = cv2.imread("picture_material/beauty_leg3.jpg")
plt.subplot(331), plt.imshow(img)
plt.axis('off'), plt.title("BGR")

img_BGR = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(332), plt.imshow(img_BGR)
plt.axis('off'), plt.title("RGB")

img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(333), plt.imshow(img_GRAY)
plt.axis('off'), plt.title("GRAY")

img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.subplot(334), plt.imshow(img_HSV)
plt.axis('off'), plt.title("HSV")

img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
plt.subplot(335), plt.imshow(img_YCrCb)
plt.axis('off'), plt.title("YCrCb")

img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
plt.subplot(336), plt.imshow(img_HLS)
plt.axis('off'), plt.title("HLS")

img_XYZ = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
plt.subplot(337), plt.imshow(img_XYZ)
plt.axis('off'), plt.title("XYZ")

img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
plt.subplot(338), plt.imshow(img_LAB)
plt.axis('off'), plt.title("LAB")

img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
plt.subplot(339), plt.imshow(img_YUV)
plt.axis('off'), plt.title("YUV")

plt.show()