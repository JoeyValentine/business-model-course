import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./img/mountain.jpg')
plt.imshow(img)
plt.show()

b_img = img[:,:,0]
g_img = img[:,:,1]
r_img = img[:,:,2]

# 이미지는 행렬이다. 이미지행렬에 전치행렬을 하면 ...
width = img.shape[0]
height = img.shape[1]
imgNew = np.zeros((height, width, 3), dtype='i')
imgNew[:,:,0] = r_img.T
imgNew[:,:,1] = g_img.T
imgNew[:,:,2] = b_img.T
plt.imshow(imgNew)
plt.show()

# 이미지 R,G,B에 평균을 하면 그레이 이미지를 얻을 수 있다.
imgGray = 0.21 * r_img + 0.72 * g_img + 0.07 * b_img
plt.imshow(imgGray, cmap='gray')
plt.show()