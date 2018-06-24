import pandas as pd
import os
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
	# 以下两个路径需要修改
	datapath = "/home/Signboard/second/datasets/train/"
	train_txt = pd.read_csv( "/home/Signboard/second/datasets/train.txt", sep=' ', names=['filepath', "label", 'xmin', 'ymin', 'xmax', 'ymax'])

	train_txt['filepath'] = datapath + train_txt['filepath']
	train, val = train_test_split(train_txt, test_size=0.1, random_state=42, stratify=train_txt['label'])
	train.to_csv("train_annotations.csv", columns=['filepath', 'xmin', 'ymin', 'xmax', 'ymax', "label"], header=False, index=False)
	val.to_csv("val_annotations.csv", columns=['filepath', 'xmin', 'ymin', 'xmax', 'ymax', "label"], header=False, index=False)

	f = open("classes.csv", "w+")
	# class_name,id
	for i in range(60):
		f.write("{},{}\n".format(i+1,i))
	f.close()