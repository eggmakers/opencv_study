import cv2

# 读取图像文件
img = cv2.imread('picture_material/location.jpg')

# 将BGR图像转换为RGB图像
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 反转颜色
inverted_img = cv2.bitwise_not(img)

# 显示图像
cv2.imshow('Inverted Image', inverted_img)
cv2.imwrite('result/inverted_image.png', inverted_img)
cv2.waitKey(0)

# 关闭窗口
cv2.destroyAllWindows()
