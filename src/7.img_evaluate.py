import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse

img_origin = cv2.imread("picture_material/beauty_leg2.jpg")
img_origin = cv2.resize(img_origin, (int(img_origin.shape[1]/2), int(img_origin.shape[0]/2)), interpolation=cv2.INTER_AREA)
cv2.imshow("Origin",img_origin)

img_Mb = cv2.imread("result/blur image.png")
cv2.imshow("Move blur image",img_Mb)

mse = compare_mse(img_Mb, img_origin)
print('MSE:{}'.format(mse))
psnr = compare_psnr(img_Mb, img_origin)
print('PSNR:{}'.format(psnr))
ssim = compare_mse(img_Mb, img_origin)
print('SSIM:{}'.format(ssim))

cv2.waitKey()