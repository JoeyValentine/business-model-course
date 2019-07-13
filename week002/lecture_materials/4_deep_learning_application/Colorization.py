from keras.activations import tanh
from tensorflow.contrib.layers import relu
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras import optimizers
from tensorflow.keras import losses

from skimage.io import imsave
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Conv2D, Dropout, Conv2DTranspose

X = np.load( './img/X.npy' )
Y = np.load( './img/Y.npy' )
test_X = np.load( './img/test_X.npy' )

print( X.shape ,Y.shape )
print( test_X.shape )

dropout_rate = 0.5
DIMEN = 64
kernel_size = ( 4 , 4 )

NEURAL_SCHEMA = [

    Conv2D( 32 , input_shape=( DIMEN , DIMEN , 1 ) , kernel_size=kernel_size , strides=1,activation=relu),
    Dropout( dropout_rate ) ,
    Conv2D( 64, kernel_size=kernel_size, strides=1, activation=relu),
    Dropout(dropout_rate),
    Conv2D( 128, kernel_size=kernel_size, strides=1, activation=relu) ,
    Dropout(dropout_rate),
    Conv2D( 256, kernel_size=kernel_size, strides=1, activation=relu),
    Dropout(dropout_rate),
    Conv2DTranspose( 128, kernel_size=kernel_size, strides=1, activation=relu),
    Dropout(dropout_rate),
    Conv2DTranspose( 64, kernel_size=kernel_size, strides=1, activation=relu),
    Dropout(dropout_rate),
    Conv2DTranspose( 32, kernel_size=kernel_size, strides=1, activation=relu),
    Dropout(dropout_rate),
    Conv2DTranspose( 3, kernel_size=kernel_size, strides=1, activation=tanh ),

]

model = tf.keras.Sequential( NEURAL_SCHEMA )

model.compile(
    optimizer=optimizers.Adam(0.0001),
    loss=losses.mean_squared_error,
    metrics=['mae'],
)
#model = models.load_model( './final_model.h5' )
model.fit(
    X,
    Y,
    batch_size=3 ,
    epochs=100,
    verbose = 2
)

values = model.predict( test_X )
values = np.maximum( values , 0 )

for i in range( 6 ):
    image_final = ( values[i] * 255).astype( np.uint8 )
    imsave( './'+'{}.png'.format( i + 1 ) , image_final  )

img_array = np.load('./img/X.npy')
print(img_array.shape)
for i in range(6):
    img = img_array[i,:,:,:]
    img = img.reshape(64,64)
    imsave('./X'+str(i)+'.png', img)

img_array = np.load('./img/Y.npy')
print(img_array.shape)
for i in range(6):
    img = img_array[i,:,:,:]
    img = img.reshape(64,64,3)
    imsave('./Y'+str(i)+'.png', img)

img_array = np.load('./img/test_X.npy')
for i in range(6):
    img = img_array[i,:,:,:]
    img = img.reshape(64,64)
    imsave('./test_X'+str(i)+'.png', img)