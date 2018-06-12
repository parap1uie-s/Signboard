import numpy as np
from PIL import Image
import os
from Model import *

# filepath - str - 绝对路径 - /home/xxx/xxx/xxx/3801213fb80e7bec39113f26262eb9389a506b29.jpg
# bboxes - list - 二维list或tuple - [ [x1, y1, x2, y2], [x1, y1, x2, y2], [x1, y1, x2, y2] ]
# nparray - bool - 返回shape为[len(bboxes), 224,224,3]的数组，或返回一组PIL.Image对象

def crop_image(filepath, bboxes, nparray=False):
    res = []
    for box in bboxes:
        assert len(box) == 4
        x1, y1, x2, y2 = box
        res.append( Image.open(filepath).crop((x1,y1,x2,y2)).resize((224,224), Image.ANTIALIAS) )
    if nparray:
        res = np.array( list(map(lambda x:np.array(x),res)) )
    return res

# def predict(model, croped_images):


def classify_main(rootpath, filepaths, all_image_bboxes):
    class_indices = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '24': 17, '25': 18, '26': 19, '27': 20, '28': 21, '29': 22, '3': 23, '30': 24, '31': 25, '32': 26, '33': 27, '34': 28, '35': 29, '36': 30, '37': 31, '38': 32, '39': 33, '4': 34, '40': 35, '41': 36, '42': 37, '43': 38, '44': 39, '45': 40, '46': 41, '47': 42, '48': 43, '49': 44, '5': 45, '50': 46, '51': 47, '52': 48, '53': 49, '54': 50, '55': 51, '56': 52, '57': 53, '58': 54, '59': 55, '6': 56, '60': 57, '7': 58, '8': 59, '9': 60}
    class_indices = dict((v,k) for k,v in class_indices.items())

    model = DenseNetTransfer((224,224,3), channel=4)
    model.load_weights("Transfer-densenet.h5", by_name=True)

    ret = []
    for k,filepath in enumerate(filepaths):
        assert os.path.isfile( os.path.join(rootpath, filepath) ), "Wrong filepath, file not founded"
        croped_images = crop_image(os.path.join(rootpath, filepath), all_image_bboxes[k], nparray=True)
        # 像素值归一化
        croped_images = croped_images / 255.0
        predict_res = model.predict(croped_images)
        label = np.argmax(predict_res, axis=1)
        label = [ class_indices[l] for l in label ]
        score = np.max(predict_res, axis=1)

        for i in range(len(label)):
            ret.append( (filepath, label[i], score[i]) )
    return ret

if __name__ == '__main__':
    # 3801213fb80e7bec63750acd232eb9389b506b6a.jpg 19 657 420 754 567
    # 3801213fb80e7bec63750acd232eb9389b506b6a.jpg 19 278 309 582 497
    # 3801213fb80e7bec451d52cb252eb9389a506bd0.jpg 6 320 286 590 594
    # 3801213fb80e7bec451d52cb252eb9389a506bd0.jpg 6 803 553 904 731
    all_image_bboxes = [ [[657,420,754,567], [278,309,582,497]], [[320,286,590,594], [803,553,904,731]] ]
    # 到包含着test图片的文件夹
    # filepath 只对应文件名，可以从test.txt中直接取
    rootpath = "/home/Signboard/second/datasets/train/"
    filepaths = ["3801213fb80e7bec63750acd232eb9389b506b6a.jpg", "3801213fb80e7bec451d52cb252eb9389a506bd0.jpg"]

    res = classify_main(rootpath, filepaths, all_image_bboxes)
    print(res)
    # [('3801213fb80e7bec63750acd232eb9389b506b6a.jpg', '19', 0.99979323), ('3801213fb80e7bec63750acd232eb9389b506b6a.jpg', '19', 0.6446284), ('3801213fb80e7bec451d52cb252eb9389a506bd0.jpg', '6', 0.9995425), ('3801213fb80e7bec451d52cb252eb9389a506bd0.jpg', '6', 0.99896395)]