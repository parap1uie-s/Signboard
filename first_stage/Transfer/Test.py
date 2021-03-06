import pandas as pd
from PIL import Image
import numpy as np
import os
from Model import *
from Utils import TestDataGen
from keras.preprocessing.image import ImageDataGenerator
import argparse

if __name__ == '__main__':
    datapath = "/home/Signboard/datasets"
    width = 448
    height = 448
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelType', required=True)
    parser.add_argument('--channel', required=False, default="4")
    args = parser.parse_args()

    assert args.modelType in ["densenet", "InceptionResNetV2", "Resnet", "xception", "inception", "nas"]
    assert args.channel.isdigit()
    args.channel = int(args.channel)

    if args.modelType == "densenet":
        model = DenseNetTransfer((height,width,3), channel=args.channel)
    elif args.modelType == "InceptionResNetV2":
        model = Transfer((height,width,3), channel=args.channel)
    elif args.modelType == "Resnet":
        model = ResNet((height,width,3), channel=args.channel)
    elif args.modelType == "xception":
        model = XceptionTransfer((height,width,3), channel=args.channel)
    elif args.modelType == "inception":
        model = InceptionTransfer((height,width,3), channel=args.channel)
    elif args.modelType == "nas":
        model = NASTransfer((height,width,3), channel=args.channel)
        
    model.load_weights("Transfer-{}.h5".format(args.modelType), by_name=True)

    class_indices = {'1': 0, '10': 1, '100': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '24': 17, '25': 18, '26': 19, '27': 20, '28': 21, '29': 22, '3': 23, '30': 24, '31': 25, '32': 26, '33': 27, '34': 28, '35': 29, '36': 30, '37': 31, '38': 32, '39': 33, '4': 34, '40': 35, '41': 36, '42': 37, '43': 38, '44': 39, '45': 40, '46': 41, '47': 42, '48': 43, '49': 44, '5': 45, '50': 46, '51': 47, '52': 48, '53': 49, '54': 50, '55': 51, '56': 52, '57': 53, '58': 54, '59': 55, '6': 56, '60': 57, '61': 58, '62': 59, '63': 60, '64': 61, '65': 62, '66': 63, '67': 64, '68': 65, '69': 66, '7': 67, '70': 68, '71': 69, '72': 70, '73': 71, '74': 72, '75': 73, '76': 74, '77': 75, '78': 76, '79': 77, '8': 78, '80': 79, '81': 80, '82': 81, '83': 82, '84': 83, '85': 84, '86': 85, '87': 86, '88': 87, '89': 88, '9': 89, '90': 90, '91': 91, '92': 92, '93': 93, '94': 94, '95': 95, '96': 96, '97': 97, '98': 98, '99': 99}

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
            os.path.join(datapath, "test_new"),
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
    gt_result = pd.read_csv( os.path.join(datapath, "test_groundtruth.csv"), sep=' ', names=['filepath', 'classid'],
        dtype={"filepath":"str", "classid":"int"})
    compare = pred_result.merge(gt_result, how='left', on='filepath')
    def equal(a, b):
        return int(a) == int(b)
    compare['res'] = compare.apply(lambda row: equal(row['classid_x'], row['classid_y']), axis=1)

    acc = compare['res'].sum()
    pred_result.to_csv("result-{:0>4}.csv".format(acc), sep=' ', header=False, index=False)