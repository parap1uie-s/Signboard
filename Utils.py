import numpy as np
from PIL import Image
import pandas as pd
import os,shutil
from keras.preprocessing.image import ImageDataGenerator

def DataGen(datapath, batch_size=32):
    data_gen_args = dict(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1/255.0,
        fill_mode='constant',
        cval=0,
        shear_range=0.2,
        zoom_range=0.2)

    train_datagen = ImageDataGenerator(**data_gen_args)
    val_datagen = ImageDataGenerator(**data_gen_args)

    train_datagen = train_datagen.flow_from_directory(
        os.path.join(datapath, "train"),
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='sparse')
    val_datagen = val_datagen.flow_from_directory(
        os.path.join(datapath, "val"),
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='sparse')
    return train_datagen, val_datagen

def moveImg(datapath):
    from sklearn.model_selection import train_test_split
    all_pd = pd.read_csv( os.path.join(datapath, "train.txt"), sep=' ', names=['filepath', 'classid'], dtype={"filepath":"str", "classid":"str"})
    for i in range(1,101):
        if not os.path.exists( os.path.join(datapath, 'train', str(i)) ):
            os.mkdir(os.path.join(datapath, 'train', str(i)))

    for i in range(1,101):
        if not os.path.exists( os.path.join(datapath, 'val', str(i)) ):
            os.mkdir(os.path.join(datapath, 'val', str(i)))   

    train_pd, val_pd = train_test_split(all_pd, test_size=0.15, random_state=111, stratify=all_pd['classid'])
    train_pd.to_csv(os.path.join(datapath, "train_split.txt"), sep=' ', header=False, index=False)
    val_pd.to_csv(os.path.join(datapath, "val_split.txt"), sep=' ',header=False, index=False)

    for row in train_pd.iterrows():
        r = row[1]
        c = r['classid']
        f = r['filepath']
        shutil.move(os.path.join(datapath, 'train', f), os.path.join(datapath, 'train', c,f))
    for row in val_pd.iterrows():
        r = row[1]
        c = r['classid']
        f = r['filepath']
        shutil.move(os.path.join(datapath, 'train', f), os.path.join(datapath, 'val', c,f))
