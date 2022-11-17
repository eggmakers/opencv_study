import cv2

img = cv2.imread("picture_material/coin.jpg")
cv2.imshow("origin", img)

blurred = cv2.GaussianBlur(img, (3, 3), 0, 0)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

edge_output = cv2.Canny(gray, 220, 250)
cv2.imshow("Canny edge", edge_output)

contours, hierarchy = cv2.findContours(edge_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    cv2.imshow("output", img)

k = cv2.waitKey()
if k == 27:
    cv2.destroyAllWindows()