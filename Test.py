import pandas as pd
from PIL import Image
import numpy as np
import os
from Model import xceptionModel

if __name__ == '__main__':
    datapath = "/home/Signboard/datasets"
    test_pd = pd.read_csv(os.path.join(datapath, "test.txt"), sep=' ', names=['filepath'], dtype={"filepath":"str"})

    model = xceptionModel((128,128,3))
    model.load_weights("w.h5", by_name=True)

    result = open("result.csv", "w+", encoding='UTF-8')

    for row in test_pd.iterrows():
        r = row[1]
        f = r['filepath']

        Img = Image.open( os.path.join(datapath,'test', f) ).resize((128, 128),Image.ANTIALIAS)
        Img = np.expand_dims(np.array( Img ),axis=0) / 255.0

        res = model.predict(Img)
        max_ind = np.argmax(res[0]).astype("int32") + 1
        result.write("{} {}\n".format(f,max_ind))