import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

base_dir = 'd:/cat_dog_smallData'
test_dir = os.path.join(base_dir, 'test')

model = load_model('Cat_Dog_VGG16_2.h5')
print(model.summary())

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        class_mode='binary',
        batch_size=1)
test_loss, test_acc = model.evaluate_generator(test_generator, steps = 1000)
print(test_acc)
