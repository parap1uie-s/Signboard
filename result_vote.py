import pandas as pd
import numpy as np
import os
import itertools

def GridMergeResult(Submission_path):
    fileLists = os.listdir(Submission_path)

    true_label = pd.read_csv("test_groundtruth.csv", sep=' ', names=['id', 'label'])['label'].values

    read_img_id = False
    labels = []
    for result in fileLists:
        if result.split(".")[-1] != "csv":
            continue
        result_handle = pd.read_csv(os.path.join(Submission_path,result),sep=' ',names=['id','label'])
        if (read_img_id == False):
            img_id = result_handle['id']
            read_img_id = True
        label = result_handle['label'].values
        labels.append(label)
    labels = np.array(labels).T

    bestAcc = 0
    bestComb = []
    for i in range(2, len(fileLists) + 1):
        for fileList in itertools.combinations(range(0,len(fileLists)), i):
            # 统计每一行次数出现最多的数字
            vote_label = []
            for value in labels[:,fileList]:
                beitai = np.unique(value)
                label_fre = [list(value).count(i) for i in beitai]
                max_label = np.argmax(np.array(label_fre))
                vote_label.append(beitai[max_label])
            vote_label = np.array(vote_label)
            tempRes = true_label == vote_label
            acc = sum(tempRes.astype(np.int64)) / len(tempRes)
            if acc >= bestAcc:
                bestAcc = acc
                bestComb = fileList
                print(bestAcc)
                print([fileLists[i] for i in bestComb])
    #生成csv文件
    print(bestAcc)
    print(bestComb)

def MergeResult(Submission_path,vote_result):
    labels = []
    # fileLists = os.listdir(Submission_path)
    # fileLists = [fileLists[i] for i in [0, 5, 6, 8, 9, 11, 12]]
    fileLists = ['result-0899-xception.csv', 'result-0984_vote.csv', 'result-0967-densenet.csv', 
    'result-0962-mutiscale_3.csv', 'result-0956-mutiscale.csv', 'result-0935-CRNN.csv', 'result-0920-resnet50.csv']

    read_img_id = False
    for result in fileLists:
        if result.split(".")[-1] != "csv":
            continue
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
    Submission_path = "Submission"

    vote_result = "vote_result.csv"
    test_truth = "test_groundtruth.csv"

    MergeResult(Submission_path,vote_result)
    Compare(vote_result,test_truth)