import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def build_generator(directory, augment=False, batch_size=8):
    if augment == True:
        data_generator = ImageDataGenerator(rotation_range=20.0,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.4,
                                           zoom_range=0.2)
    else:
        data_generator = ImageDataGenerator()

    generator = data_generator.flow_from_directory(directory=directory,
                                                        target_size=(224, 224),
                                                        batch_size=batch_size)
    return generator