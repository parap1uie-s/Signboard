环境配置：
参考https://github.com/weiliu89/caffe 配置ssd所需环境

1、标签制作（若已有lmdb文件则跳过次步骤）

   a、修改createxml中的路径，创建ssd所需的xml文件
   b、修改create_list_person和create_data_person中路径，运行生成lmdb文件

2、模型训练（按照ssd网站提供的512*512的配置文件修改）

   a、修改train.prototxt文件中的数据路径
   b、按照输出类别为61，修改模型部分参数
   c、模型共训练26w次，去bs为8
   d、修改train.sh里的路径，开始训练模型

3、模型测试

   a、进入caffe目录，运行指令build/examples/ssd/ssd_detect.bin data/baidu/VGG_512/deploy.prototxt data/baidu/VGG_512/VGG_coco_SSD_512x512_iter_260000.caffemodel test.txt -out_file=a.txt
      得到测试结果保存在当前目录中的a.txt文件中。
   b、由于ssd训练周期较长，所以保存了最后的caffemodel，可调过训练直接测试，然后将txt文件转化为csv文件，ssd单模型可达到0.8688的成绩

4、模型融合

   a、修改相关路径，运行fusion_nms.py，对已有的几个模型结果进行融合（阈值0.6），同时将超出边界框限制在边界内
   b、将txt文件，转化为csv文件
   

注意事项：我们的最终成绩融合了大量的中间结果，主要有三个模型ssd + faster rcnn + retinanet，其中ssd保存了模型以及运行结果，我们重新训练了faster rcnn和retinanet，但由于时间与显存限制，batchsize
          设置了很小的值影响了融合的最终情况。另外，融合操作均在txt文件进行，可将csv文件内容复制到同名txt文件。

任何疑问请及时联系我们！！！