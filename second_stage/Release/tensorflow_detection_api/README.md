#以下所有的操作，都基于tensorflow_detection_api，是我们根据自己的需求将官方的代码进行修改。

##backbone
faster_rcnn_resnet101

##环境安装（ubuntu16.04，NVIDIA GTX 1080，CUDA9.0.2,anaconda3）

###安装好anaconda后，创建一个tensorflow环境
```
conda create -n tensorflow
```

###激活tensorflow环境
```
source activate tensorflow
```

###安装tensorflow-gpu
```
conda install tensorflow-gpu
```

###安装依赖环境

```
sudo apt-get install protobuf-compiler python-pil python-lxml
sudo conda install jupyter,matplotlib,pillow,lxml
sudo pip install opencv-python
```

###务必需要的操作
必须编译Protobuf库，在object_detection同级目录打开终端运行：
```
protoc object_detection/protos/*.proto --python_out=.
```
若Protobuf版本不对，可以安装3.3.0版本
```
wget https://github.com/google/protobuf/archive/v3.3.0.tar.gz
tar zxvf v3.3.0.tar.gz
cd protobuf-3.3.0/
./autogen.sh
./configure
make -j
make install
```

###配置环境变量
```
vim ～/.bashrc
```
在最后加上以下语句并保存：
```
export PYTHONPATH="$PYTHONPATH:/projects/tensorflow_detection_api:/projects/tensorflow_detection_api/slim"
```


##创建数TFRecored数据集
###

###执行生成TFRecored命令
- 将9000张训练图片放到/path_to_project/tensorflow_detection_api/object_detection/voc/VOCdevkit/VOC2012/JPEGImages/下，
- 目录/path_to_project/tensorflow_detection_api/object_detection/voc/VOCdevkit/VOC2012/ImageSets/Main的文件是 --set指定的数据集列表，在训练中我使用全量数据进行训练，因此使用trainval。
- 目录/path_to_project/tensorflow_detection_api/object_detection/voc/VOCdevkit/VOC2012/Annotations下是训练训练集的xml文件
执行以下命令生成tfrecored数据集。
```
python object_detection/dataset_tools/create_pascal_tf_record.py --data_dir='voc/VOCdevkit' --label_map_path=voc/billboard_label_map.pbtxt --year=VOC2012 --set=trainval --output_path='voc/records/trainval.record'
```

##训练
数据集生成后，直接执行以下进行训练：
```
python object_detection/train.py --train_dir voc/checkpoints/ --pipeline_config_path voc/faster_rcnn_resnet101.config
```
其中faster_rcnn_resnet101.config是配置文件，由于显存限制，我们使用batch_size=1

##导出模型
参数--trained_checkpoint_prefix后的数字根据checkpoint的step数修改
```
python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path voc/faster_rcnn_resnet101.config --trained_checkpoint_prefix voc/checkpoints/model.ckpt-141606 --output_directory voc/export
```

##测试并生成提交文件
执行以下文件进行测试：
```
python object_detection/detection.py
```
相关参数配置在文件detection.py给出

