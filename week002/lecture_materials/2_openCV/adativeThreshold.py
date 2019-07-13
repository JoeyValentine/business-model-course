import cv2

img_file = './img/tomato.jpg'
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
blk_size = 9
C = 5
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk_size, C)
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()