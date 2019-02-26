# coding=utf-8
from collections import Counter
from PIL import Image
from utils import label_map_util
import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm


sys.path.append("..")

# 超参数设置
NUM_CLASSES = 60
category = "voc/categories.txt"         # 分类序号与名称映射
MODEL_NAME = 'voc/export/'              # 导出的模型位置
PATH_TO_CKPT = MODEL_NAME + 'frozen_inference_graph.pb'  # 模型路径
PATH_TO_LABELS = os.path.join('data', 'billboard_label_map.pbtxt')
SCORE_THRESHOLD = 0.02                  # 分数阈值
RESULT_CVS = "summit/result.csv"        # 输出提交的测试结果文件
DRAW_PIC = "results"                    # 画框的图片存放位置
OVERLAB = 0.95                          # 去重叠框的包含程度
PATH_TO_TEST_IMAGES_DIR = '/projects/ctpn/dataset/for_test/test'  # 源测试图片的位置

if not os.path.exists(DRAW_PIC):
    os.mkdir(DRAW_PIC)

my_dict = {}
with open(category, 'r') as fc:
    line = fc.readline()
    while line:
        cols = line.strip("\r\n").split(':')
        dict2 = {cols[0]: cols[1]}
        my_dict.update(dict2)
        line = fc.readline()

# 加载计算图和参数
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    # 读入.pb文件
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# 将神经网络检测得到的index(数字）转变为类别名
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# 函数：将图片转换为numpy数组形式
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def is_contains(box_1, box_2, h, w):
    y1_min, x1_min,  y1_max, x1_max = box_1[0]*h, box_1[1]*w, box_1[2]*h, box_1[3]*w
    y2_min, x2_min,  y2_max, x2_max = box_2[0]*h, box_2[1]*w, box_2[2]*h, box_2[3]*w
    box_1_area = (x1_max - x1_min ) *(y1_max - y1_min)
    box_2_area = (x2_max - x2_min ) *(y2_max - y2_min)

    min_area = min(box_1_area, box_2_area)

    x_lt = max(x1_min, x2_min)
    y_lt = max(y1_min, y2_min)
    x_rb = min(x1_max, x2_max)
    y_rb = min(y1_max, y2_max)

    is_contain = False
    if x_lt < x_rb and y_lt < y_rb:
        intersection_area = (x_rb - x_lt)*(y_rb -y_lt)
        overlab = min_area / intersection_area*1.0
        if overlab > OVERLAB:
            is_contain = True
    return is_contain


def compare(cboxes, h, w):
    length = len(cboxes)
    to_delete = set()
    pairs = []
    for i in range(0, length-1):
        for j in range(i+1, length):
            pairs.append((i,j))

    for (i, j) in pairs:
        if is_contains(cboxes[i], cboxes[j], h, w):
            if cboxes[i][5] > cboxes[j][5]:
                to_delete.add(j)
            else:
                to_delete.add(i)

    to_delete = list(to_delete)
    cboxes = np.delete(cboxes, to_delete, 0)
    return cboxes


def get_top_num_class(input_boxes):
    cls_arr = input_boxes[:, 4].reshape(1, input_boxes.shape[0])
    cls = input_boxes[:, 4].tolist()
    top_num_class = Counter(cls).most_common(1)[0][0]
    top_num_class_index = np.argmax(cls_arr, axis=1)
    print(top_num_class_index)
    return top_num_class


def box_filter_bak(input_boxes, h, w):
    # top_num_class = get_top_num_class(input_boxes)
    score = input_boxes[:, 5].reshape(input_boxes.shape[0], 1)
    max_score_idx = np.argmax(score, axis=0)[0]
    max_score_class = input_boxes[max_score_idx, 4]
    box_list = []

    # if max_score_class != top_num_class:
    #     print("{},{}".format(top_num_class, max_score_class))

    for i in range(input_boxes.shape[0]):
        if input_boxes[i, 5] >= SCORE_THRESHOLD and input_boxes[i, 4] == max_score_class:
            box_list.append(input_boxes[i, :])

    if len(box_list) == 0:
        box_list.append(input_boxes[max_score_idx, :])
    filted_boxes = np.array(box_list)
    filted_boxes = compare(filted_boxes, h, w)
    return filted_boxes


def box_filter(input_boxes, h, w):
    box_list = []
    for i in range(input_boxes.shape[0]):
        if input_boxes[i, 5] >= SCORE_THRESHOLD:
            box_list.append(input_boxes[i, :])
    filted_boxes = np.array(box_list)
    filted_boxes = compare(filted_boxes, h, w)
    return filted_boxes


def draw_boxes(img_path, boxes, color=(0, 0, 255)):
    img = cv2.imread(img_path)
    img_name = image_path.split('/')[-1]
    height = img.shape[0]
    width = img.shape[1]
    for box in boxes:
        y_min = box[0]*height
        x_min = box[1]*width
        y_max = box[2]*height
        x_max = box[3]*width
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color,thickness=3)
        label = str(int(box[4]))
        class_txt = "{}".format(my_dict[label])
        scores_txt = "{}".format(str(box[5])[0:6])
        cv2.putText(img, class_txt, (int(x_min), int(y_min)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, scores_txt, (int(x_min), int(y_max)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(DRAW_PIC, img_name), img)


# 获取需要检测的图片列表
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}'.format(image)) for image in
                    os.listdir(PATH_TO_TEST_IMAGES_DIR)]
# 检测代码
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        result = open(RESULT_CVS, "w+")
        for image_path in tqdm(TEST_IMAGE_PATHS):
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            height, width = image_np.shape[0], image_np.shape[1]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # 使用sess.run()开始计算
            # 根据配置文件voc.config,输出300个proposal框
            # boxes:            [1, 300, x_min, y_min, x_max, y_max]  #其中1为batch_size
            # scores:           [1, 300]                              #每一个box对应分数
            # classes:          [1, 300]                              #每一个box对应类别
            # num_detections:   [300]                                 #总共300个检测结果

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).reshape(boxes.shape[0], 1).astype(np.int32)
            scores = np.squeeze(scores).reshape(boxes.shape[0], 1)
            boxes_info = np.concatenate([boxes, classes, scores], axis=1)
            boxes_info = box_filter(boxes_info, height, height)
            draw_boxes(image_path, boxes_info, (0, 255, 0))
            image_name = image_path.split('/')[-1]
            for box in boxes_info:
                y_min = box[0] * height
                x_min = box[1] * width
                y_max = box[2] * height
                x_max = box[3] * width
                result.write("{} {} {} {} {} {} {}\n".format(image_name,
                                                             str(int(box[4])),
                                                             str(box[5]),
                                                             str(int(x_min)),
                                                             str(int(y_min)),
                                                             str(int(x_max)),
                                                             str(int(y_max))))
        result.close()


