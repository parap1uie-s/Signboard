"""
Keras RFCN
Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by parap1uie-s@github.com
"""

'''
This is a demo to Eval a RFCN model with DeepFashion Dataset
http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
'''

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras_retinanet.models import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import cv2
import pickle

def Test(model, loadpath, savepath):
    assert not loadpath == savepath, "loadpath should'n same with savepath"

    if os.path.isdir(loadpath):
        for idx, imgname in enumerate(os.listdir(loadpath)):
            if not imgname.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(imgname)
            image = read_image_bgr(os.path.join(loadpath, imgname))
            TestSinglePic(model, image, savepath, imgname)
            
    # elif os.path.isfile(loadpath):
    #     if not loadpath.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
    #         print("not image file!")
    #         return
    #     print(loadpath)
    #     img = image.img_to_array( image.load_img(loadpath) )
    #     (filename,extension) = os.path.splitext(loadpath)
    #     TestSinglePic(img, imageoriChannel, model, savepath=savepath, imgname=filename)

def TestSinglePic(model, image, savepath, imgname):
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
    boxes /= scale

    # visualize detections
    flag = False
    prev_label = -1
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if flag and score < 0.5 and prev_label != label:
            break
        flag = True
        color = label_color(label)
        
        b = box.astype(int)
        draw_box(draw, b, color=color)
        
        caption = "{} {:.3f}".format(label+1, score)
        draw_caption(draw, b, caption)
        prev_label = label

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.savefig(os.path.join(savepath, imgname),bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    parser = argparse.ArgumentParser()

    parser.add_argument('--loadpath', required=False,
                default="images/",
                metavar="evaluate images loadpath",
                help="evaluate images loadpath")
    parser.add_argument('--savepath', required=False,
            default="result/",
            metavar="evaluate images savepath",
            help="evaluate images savepath")

    args = parser.parse_args()
    model = load_model('model.h5', backbone_name='resnet50')
    Test(model, args.loadpath, args.savepath)