import pandas as pd
from PIL import Image
import numpy as np
import os
from Model import ResNet
from Utils import TestDataGen

if __name__ == '__main__':
    datapath = "/home/Signboard/datasets"
    width = 224
    height = 75
    model = ResNet((height,width,3))

    model.load_weights("w.h5", by_name=True)

    result = open("result.csv", "w+", encoding='UTF-8')

    gen = TestDataGen(datapath, height, width)
    while True:
        try:
            Img, filepath = next(gen)
            res = model.predict(Img)
            res = np.argmax(res, axis=1)
            for k,f in enumerate(filepath):
                result.write("{} {}\n".format(f,res[k]+1))
        except Exception as e:
            print(e)
            break
