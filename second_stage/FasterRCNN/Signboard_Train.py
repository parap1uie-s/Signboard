"""
Keras RFCN
Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by parap1uie-s@github.com
"""

'''
This is a demo to TRAIN a RFCN model with DeepFashion Dataset
http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
'''

from KerasRFCN.Model.Model import RFCN_Model
from KerasRFCN.Config import Config
from KerasRFCN.Utils import Dataset
import os
import pickle as pk
import numpy as np
from PIL import Image

############################################################
#  Config
############################################################

class RFCNNConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Signboard"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    C = 1 + 60  # background + 60 tags
    NUM_CLASSES = C
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 1000

    RPN_NMS_THRESHOLD = 0.7

############################################################
#  Dataset
############################################################

class SignboardDataset(Dataset):
    # count - int, images in the dataset
    def initDB(self, phase='train'):
        self.rootpath = '/home/Signboard/second/datasets/'

        # Add classes
        for i in range(1,61):
            self.add_class("Signboard",i,str(i))

        if phase == "train":
            data = pk.load( open("train.pk", "rb+") )
        else:
            data = pk.load( open("val.pk", "rb+") )
        for k, item in enumerate(data):
            self.add_image(source="Signboard",image_id=k, path=item['filepath'], width=item['width'], height=item['height'], bboxes=item['bboxes'], label=item['label'])

    # read image from file and get the 
    def load_image(self, image_id):
        info = self.image_info[image_id]
        # tempImg = image.img_to_array( image.load_img(info['path']) )
        tempImg = np.array(Image.open( os.path.join(self.rootpath, "train", info['path']) ))
        return tempImg

    def load_bbox(self, image_id):
        info = self.image_info[image_id]
        bboxes = []
        labels = []
        for item in info['bboxes']:
            bboxes.append(item)
            labels.append( info['label'] )
        return np.array(bboxes), np.array(labels)

if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    config = RFCNNConfig()
    dataset_train = SignboardDataset()
    dataset_train.initDB("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SignboardDataset()
    dataset_val.initDB("val")
    dataset_val.prepare()

    model = RFCN_Model(mode="training", config=config, model_dir=os.path.join(ROOT_DIR, "logs"),architecture='resnet101' )
    # This is a hack, bacause the pre-train weights are not fit with dilated ResNet
    # model.keras_model.load_weights("/home/professorsfx/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", 
    #     by_name=True, skip_mismatch=True)
    # model.keras_model.load_weights(COCO_MODEL_PATH, by_name=True, skip_mismatch=True)
    try:
        model_path = model.find_last()[1]
        if model_path is not None:
            model.load_weights(model_path, by_name=True)
    except Exception as e:
        print(e)
        print("No checkpoint founded")
        
    # *** This training schedule is an example. Update to your needs ***

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=160,
                layers='all')
