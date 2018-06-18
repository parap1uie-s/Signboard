import numpy as np
from PIL import Image
import os
from Model import *
import pandas as pd

def crop_image(filepath, box, width, height, nparray=False):
    res = []
    assert len(box) == 4
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
    res = Image.open(filepath).crop((x1,y1,x2,y2)).resize((width,height), Image.ANTIALIAS)
    if nparray:
        # res = np.array( list(map(lambda x:np.array(x),res)) )
        res = np.array(res)
    return res

def crop_image_generator(rootpath, filepaths, all_image_bboxes, width, height):
    i = 0
    while i < len(filepaths):
        croped_images = crop_image(os.path.join(rootpath, filepaths[i]), all_image_bboxes[i], width, height, nparray=True)
        croped_images = np.expand_dims(croped_images,axis=0) / 255.0
        i += 1
        yield croped_images


def classify_main(rootpath, filepaths, all_image_bboxes):
    class_indices = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '24': 17, '25': 18, '26': 19, '27': 20, '28': 21, '29': 22, '3': 23, '30': 24, '31': 25, '32': 26, '33': 27, '34': 28, '35': 29, '36': 30, '37': 31, '38': 32, '39': 33, '4': 34, '40': 35, '41': 36, '42': 37, '43': 38, '44': 39, '45': 40, '46': 41, '47': 42, '48': 43, '49': 44, '5': 45, '50': 46, '51': 47, '52': 48, '53': 49, '54': 50, '55': 51, '56': 52, '57': 53, '58': 54, '59': 55, '6': 56, '60': 57, '7': 58, '8': 59, '9': 60}
    class_indices = dict((v,k) for k,v in class_indices.items())
    width = 224
    height = 112
    model = DenseNetTransfer((height,width,3), channel=4)
    model.load_weights("Transfer-densenet.h5", by_name=True)
    handle = open("temp.txt", "w+")
    # ret = []
    gen = crop_image_generator(rootpath, filepaths, all_image_bboxes, width,height)
        # 像素值归一化
    predict_res = model.predict_generator(gen, steps=len(filepaths), use_multiprocessing=True, max_queue_size=100)
    label = np.argmax(predict_res, axis=1)
    label = [ class_indices[l] for l in label ]
    score = np.max(predict_res, axis=1)

    for k,l in enumerate(label):
        # if l == "0":
        #     continue
        handle.write("{} {} {}\n".format(filepaths[k], l, score[k]) )
    # return ret

if __name__ == '__main__':
    rootpath = "/home/Signboard/second/datasets/test/"

    temp_res_file = "merge.txt"
    handle_csv = pd.read_csv(temp_res_file, sep=' ', names=['filepath', "label", 'score', 'xmin', 'ymin', 'xmax', 'ymax'],
        dtype={"filepath":"str"})
    all_image_bboxes = handle_csv.loc[:,['xmin', 'ymin', 'xmax', 'ymax']].values
    classify_main(rootpath, handle_csv['filepath'].tolist(), all_image_bboxes)
    res_csv = pd.read_csv("temp.txt", sep=' ', names=['filepath', "label", 'score'], dtype={"filepath":"str", "label":"str", "score":"str"})
    handle_csv['label'] = res_csv['label']
    handle_csv['score'] = res_csv['score']

    handle_csv['xmin'] = handle_csv['xmin'].map(lambda x:str(int(round(x))) )
    handle_csv['ymin'] = handle_csv['ymin'].map(lambda x:str(int(round(x))) )
    handle_csv['xmax'] = handle_csv['xmax'].map(lambda x:str(int(round(x))) )
    handle_csv['ymax'] = handle_csv['ymax'].map(lambda x:str(int(round(x))) )
    handle_csv = handle_csv.loc[ handle_csv['label'] != '0', : ]
    handle_csv.to_csv("result.csv", sep=' ', header=False, index=False)