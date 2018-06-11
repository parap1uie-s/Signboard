import pandas as pd
from PIL import Image
import numpy as np
import os
from Model import xceptionModel
import keras.models
np.set_printoptions(suppress=True)

if __name__ == '__main__':
    datapath = "/home/Signboard/datasets"
    train_pd = pd.read_csv(os.path.join(datapath, "train_split.txt"), sep=' ', names=['filepath', 'classid'],dtype={"filepath":"str", "classid":"str"})
    val_pd = pd.read_csv(os.path.join(datapath, "val_split.txt"), sep=' ', names=['filepath', 'classid'],dtype={"filepath":"str", "classid":"str"})
    test_pd = pd.read_csv(os.path.join(datapath, "test.txt"), sep=' ', names=['filepath'], dtype={"filepath":"str"})

    model = xceptionModel((139,139,3))
    model = keras.models.Model(inputs=model.input, outputs=model.get_layer("dense_1").output)
    model.load_weights("w.h5", by_name=True)

    result = open("train_features.csv", "w+", encoding='UTF-8')

    for row in train_pd.iterrows():
        r = row[1]
        f = r['filepath']
        label = r['classid']

        Img = Image.open( os.path.join(datapath,'train', label, f) ).resize((139, 139),Image.ANTIALIAS)
        Img = np.expand_dims(np.array( Img ),axis=0) / 255.0

        features = model.predict(Img)[0].astype(str).tolist()
        result.write( ",".join(features)+",{}\n".format(label) )


    result = open("val_features.csv", "w+", encoding='UTF-8')
    for row in val_pd.iterrows():
        r = row[1]
        f = r['filepath']
        label = r['classid']

        Img = Image.open( os.path.join(datapath,'val', label, f) ).resize((139, 139),Image.ANTIALIAS)
        Img = np.expand_dims(np.array( Img ),axis=0) / 255.0

        features = model.predict(Img)[0].astype(str).tolist()
        result.write( ",".join(features)+",{}\n".format(label) )


    result = open("test_features.csv", "w+", encoding='UTF-8')
    for row in test_pd.iterrows():
        r = row[1]
        f = r['filepath']

        Img = Image.open( os.path.join(datapath,'test', f) ).resize((139, 139),Image.ANTIALIAS)
        Img = np.expand_dims(np.array( Img ),axis=0) / 255.0

        features = model.predict(Img)[0].astype(str).tolist()
        result.write( ",".join(features)+"\n" )