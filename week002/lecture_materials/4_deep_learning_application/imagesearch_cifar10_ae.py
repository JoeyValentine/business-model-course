import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import datasets
import matplotlib.pyplot as plt
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = np.reshape(X_train, (-1, 32, 32, 3))
X_test = np.reshape(X_test, (-1, 32, 32, 3))

num_classes = 10
batch_size = 32
print(X_train.shape, X_test.shape)

input_img = Input(shape=(32, 32, 3))
x = Conv2D(16,(3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same', name='encoder')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()
"""## Train it"""

autoencoder.fit(X_train, X_train, epochs=2, batch_size=32, callbacks=None, verbose=2)

autoencoder.save('autoencoder.h5')

encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
encoder.save('encoder.h5')
query = X_test[7]
plt.imshow(query.reshape(32,32,3))

X_test = np.delete(X_test, 7, axis=0)
print(X_test.shape)

codes = encoder.predict(X_test)
query_code = encoder.predict(query.reshape(1,32, 32, 3))

from sklearn.neighbors import NearestNeighbors
n_neigh = 5

codes = codes.reshape(-1, 4*4*8); print(codes.shape)
query_code = query_code.reshape(1, 4*4*8); print(query_code.shape)

"""### Fit the KNN to the test set"""

nbrs = NearestNeighbors(n_neighbors=n_neigh).fit(codes)

distances, indices = nbrs.kneighbors(np.array(query_code))

closest_images = X_test[indices]

closest_images = closest_images.reshape(-1,32, 32, 3); print(closest_images.shape)

plt.imshow(query.reshape(32, 32, 3))

plt.figure(figsize=(20, 6))
for i in range(n_neigh):
    # display original
    ax = plt.subplot(1, n_neigh, i+1)
    plt.imshow(closest_images[i].reshape(32, 32, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

