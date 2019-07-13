from __future__ import print_function
import keras

from keras.models import load_model
from keras.datasets import cifar10
import numpy as np

num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True
n = 3

# Load the CIFAR10 data.
(_,_), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_test.shape[1:]

# Normalize data.
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_test_mean = np.mean(x_test, axis=0)
    x_test -= x_test_mean

# Convert class vectors to binary class matrices.
y_test = keras.utils.to_categorical(y_test, num_classes)

model = load_model('Cifar10_RESNET')
model.summary()

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])