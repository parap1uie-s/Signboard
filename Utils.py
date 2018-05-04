import numpy as np
from PIL import Image
import pandas as pd
import os,shutil
from keras.preprocessing.image import ImageDataGenerator
import random

def DataGen(datapath, batch_size=32, phase='train'):
    assert phase in ['train','val']
    if phase == 'train':
        csv_handle = pd.read_csv(os.path.join(datapath, "train_split.txt"), sep=' ', names=['filepath', 'classid'],dtype={"filepath":"str", "classid":"str"})
    else:
        csv_handle = pd.read_csv(os.path.join(datapath, "val_split.txt"), sep=' ', names=['filepath', 'classid'],dtype={"filepath":"str", "classid":"str"})

    data_num = len(csv_handle)

    while True:
        x = []
        y = []
        ind = np.random.choice(data_num, batch_size, replace=False)
        choiced_data = csv_handle.iloc[ind,:]
        for row in choiced_data.iterrows():
            r = row[1]
            Img = Image.open(os.path.join(datapath, phase, r['classid'], r['filepath'])).resize((128,128),Image.ANTIALIAS)

            aug =  random.randint(0,2)
            # 水平翻转
            if aug == 1:
                Img = Img.transpose(Image.FLIP_LEFT_RIGHT)
            elif aug == 2:
                Img = Img.transpose(Image.FLIP_TOP_BOTTOM)
            x.append(np.array(Img))
            y.append(int(r['classid'])-1)

        x = np.array(x) / 255.0
        y = np.array(y)
        yield x, y

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
