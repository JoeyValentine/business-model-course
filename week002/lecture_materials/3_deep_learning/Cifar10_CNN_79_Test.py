import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras import datasets
from keras.utils import np_utils
from keras.models import load_model

(_,_), (x_test, y_test) = datasets.cifar10.load_data()
Y_test = np_utils.to_categorical(y_test)

img_rows, img_cols, _ = x_test.shape[1:]

X_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)

X_test = X_test.astype('float32')

X_test /= 255

model = load_model('Cifar10_CNN_79.h5')
print(model.summary())
score = model.evaluate(X_test, Y_test)
print()
print('Test loss:', score[0])
print('Test accuracy:', score[1])
