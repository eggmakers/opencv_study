from skimage import util
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("picture_material/beauty_leg2.jpg")
img = np.array(img)

noise_gs_img = util.random_noise(img, mode = 'gaussian')        #高斯噪声
noise_salt_img = util.random_noise(img, mode = 'salt')          #盐噪声
noise_pepper_img = util.random_noise(img, mode = 'pepper')      #胡椒噪声 
noise_sp_img = util.random_noise(img, mode = 's&p')              #椒盐噪声
noise_speckle_img = util.random_noise(img, mode = 'speckle')    #乘性噪声

plt.subplot(2, 3, 1),plt.title('Original')
plt.axis('off'),plt.imshow(img)
plt.subplot(2, 3, 2),plt.title('salt')
plt.axis('off'),plt.imshow(noise_salt_img)
plt.subplot(2, 3, 3),plt.title('pepper')
plt.axis('off'),plt.imshow(noise_pepper_img)
plt.subplot(2, 3, 4),plt.title('sp')
plt.axis('off'),plt.imshow(noise_sp_img)
plt.subplot(2, 3, 5),plt.title('speckle')
plt.axis('off'),plt.imshow(noise_speckle_img)
plt.subplot(2, 3, 6),plt.title('gaussian')
plt.axis('off'),plt.imshow(noise_gs_img)
plt.show()