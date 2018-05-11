import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import os
import pandas as pd
from PIL import Image
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

def TestDataGen(datapath, height, width, batch_size=32):
    csv_handle = pd.read_csv(os.path.join(datapath, "test.txt"), sep=' ', names=['filepath'],dtype={"filepath":"str"})

    data_num = len(csv_handle)
    ind = 0
    while True:
        x = []
        y = []
        if ind >= data_num:
            break
        choiced_data = csv_handle.iloc[ind:min(ind+batch_size,data_num),:]
        for row in choiced_data.iterrows():
            r = row[1]
            Img = Image.open(os.path.join(datapath, 'test', r['filepath'])).resize((width,height),Image.ANTIALIAS)
            x.append(np.array(Img))
            y.append(r['filepath'])

        x = np.array(x) / 255.0
        y = np.array(y)
        ind += batch_size
        yield x, y