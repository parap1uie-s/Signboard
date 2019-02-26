# 百度·西交大 大数据竞赛2018 —— 商家招牌识别

## 队伍简介：团队名：parapluie，0.879,14/100（截至2018/7/13）

## 项目说明

复赛使用的三种模型之一，Retinanet

模型实现源自[github](https://github.com/fizyr/keras-retinanet)

### 运行环境

- python 3.6.5

- tensorflow-gpu 1.8.0

- keras 2.2.0

- keras-resnet 0.1.0

- Numpy 1.14.3

- Pandas 0.23.0

- Cython 0.28.4

### 数据预处理与数据清洗

- 更正train.txt中坐标值为负的部分坐标

- 使用makeData.py基于官方数据构造本模型项目使用的数据格式

### 模型介绍

- 以迁移学习为主，结合图像特性修改模型结构，使用特征工程等竞赛技巧进行优化。

- 作为参与结果融合的模型之一，我们在Retinanet这里使用了resnet101的backbone与mscoco的预训练权重（github项目release中提供）

### 代码介绍及Usage

Note：虽然我们提供了训练权值及相关训练、测试代码，但由于我们划分的val源自一次随机划分。此外batch-size及图像resize尺寸的修改，这些均可能导致在原始数据下重新训练和预测的精度与我们的线上成绩有所出入。

#### 总览

准备代码运行环境：``` python3 setup.py build_ext --inplace ```

可直接执行train.sh，包含自动下载预训练权重，自动执行makeData.py，自动训练。

但需要提前修改makeData.p中的硬编码路径

#### 数据构造

- Usage: ```python3 makeData.py```

- 使用说明：需要修改脚本中涉及到的两个硬编码路径，适配运行环境中train.txt与train图片的文件夹路径。

- 生成结果说明：9:1随机划分train/val，生成train_annotation.csv与val_annotation.csv，以及包含类别id->类别名称映射关系的classes.csv

#### 训练

- Usage: ```python keras_retinanet/bin/train.py --backbone=resnet101 --weights=resnet50_coco_best_v2.1.0.h5 --multi-gpu=3 --multi-gpu-force --image-min-side=800 --image-max-side=800 --batch-size=8 --steps=2000 csv train_annotations.csv classes.csv --val-annotations=val_annotations.csv```

- 训练说明：需提前下载好预训练权重resnet50_coco_best_v2.1.0.h5，并放置在整个项目的根目录下（与readme同级）

- 训练说明：我们使用了三个callback来确保模型达到最佳拟合状态而没有过拟合：EarlyStopping,ModelCheckpoint,LearningRateScheduler，并使用数据扩增

#### 预测

- 转换模型 ```python keras_retinanet/bin/convert_model.py snapshots/resnet101_csv_35.h5 model.h5```

- 说明：将缓存文件中训练阶段的模型权重，加载到预测结构的模型中，并生成model.h5文件

- Usage ```python predict_result.py --loadpath=xxx```

- 参数说明：loadpath指定datasets/test路径，即包含所有测试图片的路径

- 预测说明：正确设定loadpath数据路径后，在当前目录下生成一个result.csv