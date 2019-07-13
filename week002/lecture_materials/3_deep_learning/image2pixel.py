# image2pixel.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
filename = "./img/woman.jpg"
image = mpimg.imread(filename)
x = tf.Variable(image, name='x')
print(x)
model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    result = session.run(x)
    print(result.shape)
    print(result)
    plt.imshow(result)
    plt.show()
