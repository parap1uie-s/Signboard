import pandas as pd
import numpy as np
import os

def MergeResult(Submission_path,vote_result):
    labels = []
    num_file = len(os.listdir(Submission_path))
    read_img_id = False
    for result in os.listdir(Submission_path):
        result_handle = pd.read_csv(os.path.join(Submission_path,result),sep=' ',names=['id','label'])
        if (read_img_id == False):
            img_id = result_handle['id']
            read_img_id = True
        label = result_handle['label'].values
        labels.append(label)
    labels = np.array(labels).T

    # 统计每一行次数出现最多的数字
    vote_label = []
    for value in labels:
        beitai = np.unique(value)
        label_fre = [list(value).count(i) for i in beitai]
        max_label = np.argmax(np.array(label_fre))
        vote_label.append(beitai[max_label])
    vote_label = np.array(vote_label)

    #生成csv文件

    sub = pd.DataFrame({'id': img_id, 'lable': vote_label})
    sub.to_csv(vote_result, index=False, header=False, sep=' ')

def Compare(vote_result,test_truth):
    new_label = pd.read_csv(vote_result, sep=' ', names=['id', 'label'])['label'].values
    true_label = pd.read_csv(test_truth, sep=' ', names=['id', 'label'])['label'].values

    result = true_label == new_label
    acc = sum(result.astype(np.int64)) / len(result)

    print('acc of vote_result',acc)

if __name__ == '__main__':
    Submission_path = "E:\\shangfangxin\\Git\\Signboard\\Submission"

    vote_result = "vote_result.csv"
    test_truth = "MakeGT/test_groundtruth.csv"
    MergeResult(Submission_path,vote_result)
    Compare(vote_result,test_truth)


