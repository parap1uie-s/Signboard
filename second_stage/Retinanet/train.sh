#!/bin/bash
if [ ! -d "resnet50_coco_best_v2.1.0.h5" ];then
wget https://github.com/fizyr/keras-retinanet/releases/download/0.3.1/resnet50_coco_best_v2.1.0.h5
fi

python3 makeData.py
sudo python3 keras_retinanet/bin/train.py --weights=snapshots/resnet50_csv_31.h5 --multi-gpu=2 --batch-size=4 --multi-gpu-force --steps=1000 csv train_annotations.csv classes.csv --val-annotations=val_annotations.csv

sudo python3 keras_retinanet/bin/convert_model.py snapshots/resnet50_csv_31.h5 model.h5

sudo python3 keras_retinanet/bin/train.py --backbone=resnet101 --weights=snapshots/resnet101_csv_08.h5 --multi-gpu=2 --multi-gpu-force --batch-size=4 --steps=1000 csv train_annotations.csv classes.csv --val-annotations=val_annotations.csv

sudo python3 keras_retinanet/bin/train.py --weights=resnet50_coco_best_v2.1.0.h5 --multi-gpu=2 --multi-gpu-force --batch-size=4 --steps=2000 csv train_annotations.csv classes.csv --val-annotations=val_annotations.csv


python3 keras_retinanet/bin/train.py --weights=resnet50_coco_best_v2.1.0.h5 --batch-size=2 --steps=1000 csv train_annotations.csv classes.csv --val-annotations=val_annotations.csv