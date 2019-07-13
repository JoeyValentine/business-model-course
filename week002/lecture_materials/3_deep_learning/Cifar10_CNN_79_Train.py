import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras import datasets
from keras import layers, models
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

print(x_train.shape[1:])
img_rows, img_cols, _ = x_train.shape[1:]

X_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
X_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(Y_train[0])
num_classes = 10
batch_size = 32
print(input_shape)

x = layers.Input(shape=input_shape,  name='input')
h = layers.BatchNormalization()(x)
h = layers.Conv2D(32, kernel_size=(3, 3), activation='relu',  name='conv1')(h)
h = layers.Dropout(0.2)(h)
h = layers.BatchNormalization()(h)
h = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(h)
h = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(h)
h = layers.BatchNormalization()(h)
h = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(h)
h = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(h)
h = layers.Flatten()(h)
h = layers.Dropout(0.2)(h)
h = layers.BatchNormalization()(h)
h = layers.Dense(512, activation='relu', name='hidden')(h)
h = layers.Dropout(0.2)(h)
h = layers.BatchNormalization()(h)
y = layers.Dense(num_classes, activation='softmax', name='output')(h)

model = models.Model(x, y)
print(model.summary())
epochs = 25
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=batch_size,
          epochs=epochs, validation_split=0.1, verbose=2)
model.save('Cifar10_CNN_79.h5')
score = model.evaluate(X_test, Y_test)
print()
print('Test loss:', score[0])
print('Test accuracy:', score[1])
