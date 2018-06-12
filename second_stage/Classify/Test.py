import pandas as pd
from PIL import Image
import numpy as np
import os
from Model import *
from Utils import TestDataGen
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
	# 修改成你自己的文件路径
    datapath = "/home/Signboard/datasets/croped_test/"
    width = 224
    height = 224

    model = DenseNetTransfer((height,width,3), channel=4)
        
    model.load_weights("Transfer-densenet.h5", by_name=True)

    class_indices = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '24': 17, '25': 18, '26': 19, '27': 20, '28': 21, '29': 22, '3': 23, '30': 24, '31': 25, '32': 26, '33': 27, '34': 28, '35': 29, '36': 30, '37': 31, '38': 32, '39': 33, '4': 34, '40': 35, '41': 36, '42': 37, '43': 38, '44': 39, '45': 40, '46': 41, '47': 42, '48': 43, '49': 44, '5': 45, '50': 46, '51': 47, '52': 48, '53': 49, '54': 50, '55': 51, '56': 52, '57': 53, '58': 54, '59': 55, '6': 56, '60': 57, '7': 58, '8': 59, '9': 60}
    class_indices = dict((v,k) for k,v in class_indices.items())
    result = {}

    # gen = TestDataGen(datapath, height, width)
    # while True:
    #     try:
    #         Img, filepath = next(gen)
    #         res = model.predict(Img)
    #         res = np.argmax(res, axis=1)
    #         for k,f in enumerate(filepath):
    #             result[f.split("/")[-1]] = class_indices[res[k]]
    #     except Exception as e:
    #         print(e)
    #         break

    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(
            os.path.join(datapath, "test"),
            target_size=(height, width),
            shuffle = False,
            class_mode=None,
            batch_size=1,
            follow_links=True)

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    res = model.predict_generator(test_generator,steps = nb_samples)
    res = np.argmax(res, axis=1)

    for k,file in enumerate(filenames):
        result[file.split("/")[1]] = class_indices[res[k]]

    pred_result = pd.DataFrame.from_dict(result,orient='index').reset_index()
    pred_result.columns = ['filepath', 'classid']
    pred_result.to_csv("result.csv", sep=' ', header=False, index=False)