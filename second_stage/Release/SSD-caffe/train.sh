cd /home/zgp/ssd/caffe
./build/tools/caffe train \
--solver="data/baidu/VGG_512/solver.prototxt" \
--weights="data/baidu/VGG_512/VGG_coco_SSD_512x512.caffemodel" \
--gpu 0 2>&1 | tee data/baidu/VGG_BAIDU_SSD_512.log
