# 在该文件中使用的box顺序为x_min, y_min, x_max, y_max

import numpy as np

# 计算两个框的iou
def iou(coordinate1, coordinate2):

    x1_min, y1_min, x1_max, y1_max = coordinate1
    x2_min, y2_min, x2_max, y2_max = coordinate2
    # 计算交集坐标
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    # 交集边框的长和宽
    w = max(0, abs(xi_max-xi_min))
    l = max(0, abs(yi_max-yi_min))
    # 此时两个框没有交集
    if w == 0 or l == 0:
        return 0
    # 交集面积
    intersection = w*l

    # 两个框的面积
    square1 = abs(x1_max-x1_min)*abs(y1_max-y1_min)
    square2 = abs(x2_max-x2_min)*abs(y2_max-y2_min)

    # 并集面积
    union = square1 + square2 - intersection
    
    return intersection*1.0/union

# x为阈值
# boxes.shape = (N, 4)
# scores.shape = (N, 1) or (N, )
# labels.shape = (N, 1) or (N, )
def boxes_filter(boxes, scores, labels, iou_threshold=0.5):
    boxes = boxes.reshape(-1,4)
    scores = scores.reshape(-1,1)
    labels = labels.reshape(-1,1)
    # 将数据合并，方便排序
    data = np.concatenate((boxes, scores, labels), axis=1)
    
    # 对数组按照x1进行排序
    # data = np.sort(data, axis=0, kind='quicksort', order=None)

    # 存储要删除的行数
    delete_rows = []

    # 先过滤掉坐标为0的点
    for row in range(data.shape[0]):
        if data[row][2] - data[row][0] <= 1.0 or data[row][3] - data[row][1] <= 1.0:
            delete_rows.append(row)
    data = np.delete(data, delete_rows, axis=0)

    # 重置delete_rows
    delete_rows = []

    for row1 in range(data.shape[0]):

        # 假设row1已经在删除列表中，直接跳过
        if row1 in delete_rows:
            continue

        for row2 in range(row1+1, data.shape[0]):

            # 假设row2已经在删除列表中，直接跳过
            if row2 in delete_rows:
                continue

            # 此时两个框没有交集
            # if data[row2][0] >= data[row1][2]:
            #     continue
            # 当iou大于阈值，留下置信度较高的那个
            if iou(data[row1][0:4], data[row2][0:4]) > iou_threshold:
                if data[row1][-2] > data[row2][-2]:
                    delete_rows.append(row2)
                else:
                    delete_rows.append(row1)
                    # break

    data = np.delete(data, delete_rows, axis=0)

    return data[:,0:4], data[:, 4], data[:, 5]

# with open('data.pk', 'rb') as file:
#     data = pickle.load(file)

#     boxes = data[0].reshape(-1,4)
#     scores = data[1].reshape(-1,1)
#     labels = data[2].reshape(-1,1)

# data = boxes_filter(boxes, scores, labels, 0.5)
# np.savetxt('process_boxes.txt', data)