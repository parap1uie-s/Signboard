import pandas as pd
import os
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
	# 以下两个路径需要修改
	# datapath = "/home/Signboard/second/datasets/train/"
	# train_txt = pd.read_csv( "/home/Signboard/second/datasets/train.txt", sep=' ', names=['filepath', "label", 'xmin', 'ymin', 'xmax', 'ymax'])

	datapath = "/home/houjun/projects/Retinanet/datasets/train/"
	train_txt = pd.read_csv( "/home/houjun/projects/Retinanet/datasets/train.txt", sep=' ', names=['filepath', "label", 'xmin', 'ymin', 'xmax', 'ymax'])

	all_images = train_txt['filepath'].value_counts().reset_index()
	all_images.columns = ["filepath", 'counts']
	all_images = all_images.merge(train_txt[['filepath','label']], on='filepath').drop_duplicates().reset_index(drop=True)

	train_image, val_image = train_test_split(all_images, test_size=0.1, random_state=42, stratify=all_images['label'])

	train = train_image.merge(train_txt, on='filepath', how='left')
	val = val_image.merge(train_txt, on='filepath', how='left')

	print(train.shape)
	print(val.shape)
	train['filepath'] = datapath + train['filepath']
	val['filepath'] = datapath + val['filepath']

	train.to_csv("train_annotations.csv", columns=['filepath', 'xmin', 'ymin', 'xmax', 'ymax', "label_x"], header=False, index=False)
	val.to_csv("val_annotations.csv", columns=['filepath', 'xmin', 'ymin', 'xmax', 'ymax', "label_x"], header=False, index=False)

	f = open("classes.csv", "w+")
	# class_name,id
	for i in range(60):
		f.write("{},{}\n".format(i+1,i))
	f.close()
