import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

pic = plt.imread('./img/1.jpeg')/255  # dividing by 255 to bring the pixel values between 0 and 1
print(pic.shape)
plt.imshow(pic)

pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
print(pic_n.shape)

kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)
plt.show()