import cv2

img = cv2.imread("result/Origin4.png")
dst = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imshow("origin", dst)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#二值化
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
cv2.imshow("Binary", binary)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
binary_zero = cv2.erode(binary, kernel)
cv2.imshow("Binary erode img", binary_zero)

contours, heriachy = cv2.findContours(binary_zero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, contours in enumerate(contours):
    cv2.drawContours(img, contours, i, (255, 0, 255), 2, -1)
    cv2.imshow("Contours img", img)

k = cv2.waitKey()
if k == 27:
    cv2.destroyAllWindows()