import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#import  data
im = cv2.imread('./img/image.jpg')

#reshape data
reshape_data = im.reshape((im.shape[0]*im.shape[1],im.shape[2]))

#import kmeans
kmeans = KMeans(n_clusters=10, random_state=0)

#fit kmeans
kmeans.fit(reshape_data)

#data transform
image_cluster = kmeans.predict(reshape_data).reshape((im.shape[0],im.shape[1]))

#imshow
# plt.imshow(image_cluster)
# plt.show()

for i in range(8):
    crop = (image_cluster == i)
    imgR = im[:,:,0] * crop
    imgG = im[:,:,1] * crop
    imgB = im[:,:,2] * crop

    rgbArray = np.zeros(im.shape, 'uint8')
    rgbArray[:,:, 0] = imgB
    rgbArray[:,:, 1] = imgG
    rgbArray[:,:, 2] = imgR
    cropImage = Image.fromarray(rgbArray)
    plt.imshow(cropImage)
    plt.show()