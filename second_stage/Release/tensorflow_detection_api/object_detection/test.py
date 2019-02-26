import numpy as np

def is_contains(box_1, box_2):
    y1_min, x1_min,  y1_max, x1_max = box_1[0]*1280, box_1[1]*960, box_1[2]*1280, box_1[3]*960
    y2_min, x2_min,  y2_max, x2_max = box_2[0]*1280, box_2[1]*960, box_2[2]*1280, box_2[3]*960
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
        if overlab > 0.9:
            is_contain = True
    return is_contain


def compare(boxes):
    length = len(boxes)
    boxes_copy = boxes
    to_delete = set()
    for i in range(0, length-1):
        for j in range(i+1, length):
            if is_contains(boxes[i], boxes_copy[j]):
                if boxes[i][5] > boxes_copy[j][5]:
                    to_delete.add(j)
                else:
                    to_delete.add(i)
    for idx in to_delete:
        boxes = np.delete(boxes, idx, 0)
    return boxes
