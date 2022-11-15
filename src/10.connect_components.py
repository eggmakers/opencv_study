import cv2
import numpy as np

src = cv2.imread("picture_material/beauty_leg2.jpg")
src = cv2.resize(src, ((int(src.shape[1]/2), int(src.shape[0]/2))), interpolation=cv2.INTER_AREA)
src = cv2.GaussianBlur(src, (3, 3), 0)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("origin", src)
cv2.imshow("binary", binary)

output = cv2.connectedComponents(binary, connectivity = 8, ltype = cv2.CV_32S)
num_labels = output[0]
labels = output[1]

colors = [ ]
for i in range(num_labels):
    b = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    r = np.random.randint(0, 256)
    colors.append((b, g, r))
colors[0] = (0, 0, 0)

#画出连通图
h, w = gray.shape
image = np.zeros((h, w, 3), dtype = np.uint8)
for row in range(h):
    for col in range(w):
        image[row, col] = colors[labels[row, col]]
        
cv2.imshow("colors labels", image)
print("total components : ", image)
cv2.waitKey()