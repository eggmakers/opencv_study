import cv2
from skimage import util

img = cv2.imread("picture_material/beauty_leg2.jpg")
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
img = util.random_noise(img, mode = 's&p')              #椒盐噪声

dst1 = cv2.boxFilter(img, -1, (3, 3), normalize = 1)
dst2 = cv2.boxFilter(img, -1, (2, 2), normalize = 0)

cv2.imshow('Origin', img)
cv2.imshow('n = 1', dst1)
cv2.imshow('n = 0', dst2)
cv2.waitKey()