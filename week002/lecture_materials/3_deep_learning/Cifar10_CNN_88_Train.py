import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras import datasets
from keras import layers, models
from keras.utils import np_utils
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

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
h = layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                  kernel_regularizer=regularizers.l2(1.E-04), name='conv1')(h)
h = layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                  kernel_regularizer=regularizers.l2(1.E-04), name='conv2')(h)
h = layers.BatchNormalization()(h)
h = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(h)
h = layers.Dropout(0.2)(h)

h = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(1.E-04), name='conv3')(h)
h = layers.BatchNormalization()(h)
h = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(1.E-04), name='conv4')(h)
h = layers.BatchNormalization()(h)
h = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(h)
h = layers.Dropout(0.3)(h)
h = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(1.E-04), name='conv5')(h)
h = layers.BatchNormalization()(h)
h = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(1.E-04), name='conv6')(h)
h = layers.BatchNormalization()(h)
h = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(h)
h = layers.Dropout(0.4)(h)

h = layers.Flatten()(h)
y = layers.Dense(num_classes, activation='softmax', name='output')(h)

model = models.Model(x, y)
print(model.summary())
epochs = 25

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003
    return lrate

datagen = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(X_train)
print(x_train.shape[0] // batch_size)
epochs = 125
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=2000, epochs=epochs, verbose=2,
                    validation_data=(X_test, Y_test), callbacks=[LearningRateScheduler(lr_schedule)])
model.save('Cifar10_CNN_88.h5')
score = model.evaluate(X_test, Y_test)
print()
print('Test loss:', score[0])
print('Test accuracy:', score[1])

