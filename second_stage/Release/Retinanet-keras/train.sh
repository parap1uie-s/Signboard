#!/bin/bash
if [ ! -d "resnet50_coco_best_v2.1.0.h5" ];then
wget https://github.com/fizyr/keras-retinanet/releases/download/0.3.1/resnet50_coco_best_v2.1.0.h5
fi

python makeData.py
python keras_retinanet/bin/train.py --backbone=resnet101 --weights=resnet50_coco_best_v2.1.0.h5 --multi-gpu=3 --multi-gpu-force --image-min-side=800 --image-max-side=800 --batch-size=32 --steps=2000 csv train_annotations.csv classes.csv --val-annotations=val_annotations.csv