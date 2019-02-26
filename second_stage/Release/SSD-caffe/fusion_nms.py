from PIL import Image
import matplotlib.pyplot as plt
import math
import os
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import cv2


root_image_dir = 'D:/Thunder/downloads/datasets_fusai/datasets/'

def nms(dets, thresh):
    """Pure Python NMS baseline."""

    x1 = dets[:, 2]
    y1 = dets[:, 3]
    x2 = dets[:, 4]
    y2 = dets[:, 5]
    scores = dets[:, 1]


    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])


        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)


        inds = np.where(ovr <= thresh)[0]

        order = order[inds + 1]

    return keep


def fusion():
    f = open('E:/fusion1/ssd.txt', 'r')
    f1 = open('E:/fusion1/fusion.txt', 'r')
    f2 = open('E:/fusion1/final.txt', 'w')

    outcome = defaultdict(list)

    for i in open('E:/fusion1/ssd.txt'):
        line = f.readline().strip('\n')
        x = line.split('.jpg ')
        outcome[x[0]].append(x[1])

    for i in open('E:/fusion1/fusion.txt'):
        line = f1.readline().strip('\n')
        x = line.split('.jpg ')
        outcome[x[0]].append(x[1])


    for key in outcome:
        boxs = []
        for prediction in outcome[key]:
            boxs.append(prediction.split(' '))
        boxs_ = np.array(boxs, dtype= float)
        boxs_nms = nms(boxs_, 0.6)

        for i in boxs_nms:
            f2.write('aaa/' + key + '.jpg ' + boxs[i][0] + ' ' + boxs[i][1] + ' '+ boxs[i][2] + ' ' + boxs[i][3] + ' ' + boxs[i][4] + ' ' + boxs[i][5] + '\n' )

    f.close()
    f1.close()
    f2.close()

def final_label():
    f = open('E:/final.txt','r')

    f6 = open('E:/final_.txt', 'w')

    outcome = defaultdict(list)

    for i in open('E:/final.txt'):
        line = f.readline().strip('\n')
        x = line.split('/')
        y = x[-1].split('.jpg ')
        outcome[y[0]].append(y[1])

    for key in outcome:

        im = cv2.imread(root_image_dir + 'test/' + key + '.jpg')

        img_size = im.shape
        width = img_size[1]
        height = img_size[0]

        # print width, height

        for j in outcome[key]:
            #print j

            y = j.split(' ')
            #print y

            if (int(y[2]) < 0):
                y[2] = '0'

            if (int(y[3]) < 0):
                y[3] = '0'

            if (int(y[4]) > width):
                y[4] = str(width)

            if (int(y[5]) > height):
                y[5] = str(height)

            f6.write('aaa/' + key + '.jpg ' + y[0] + ' ' + y[1] + ' '+ y[2] + ' ' + y[3] + ' ' + y[4] + ' ' + y[5] + '\n')
    f.close()
    f6.close()





if __name__ =='__main__':
    fusion()
    final_label()
