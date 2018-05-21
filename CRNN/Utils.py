import numpy as np
from PIL import Image
import pandas as pd
import os,shutil
from keras.preprocessing.image import ImageDataGenerator
import random

def TestDataGen(datapath, width, height, batch_size=32):
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

            aug = random.randint(0,2)
            # 水平翻转
            if aug == 1:
                Img = Img.transpose(Image.FLIP_LEFT_RIGHT)
            elif aug == 2:
                Img = Img.transpose(Image.FLIP_TOP_BOTTOM)
            x.append(np.array(Img))
            y.append(r['filepath'])

        x = np.array(x) / 255.0
        y = np.array(y)
        ind += batch_size
        yield x, y