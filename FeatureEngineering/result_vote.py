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

def MergeResult(Submission_path):
    true_label = pd.read_csv("test_groundtruth.csv", sep=' ', names=['id', 'label'])['label'].values
    labels = []
    # 0987
    # fileLists = ['result-0985_vote.csv', 'result-0899-xception.csv', 'result-0920-resnet50.csv', 
    # 'result-0941-xception.csv', 'result-0956-mutiscale.csv', 'result-0962-mutiscale_3.csv', 'result-0972-densenet_4c.csv']
    # 0989
    # fileLists = ['result-0987_vote.csv', 'result-0962-mutiscale_3.csv', 'result-0967-densenet.csv', 
    # 'result-0972-densenet_4c.csv', 'result-0956-mutiscale.csv', 'result-0941-xception.csv', 'result-0938-CRNN_4c.csv']
    # 0990
    # fileLists = ['result-0958-xception_4c.csv', 'result-0987_vote.csv', 'result-0989_vote.csv', 'result-0972-densenet_4c.csv', 'result-0958-mutiscale.csv']
    # 0991
    # fileLists = ['result-0962-inception_resv2_4c_aug.csv', 'result-0987_vote.csv', 'result-0985_vote.csv', 'result-0945-resnet50_STN4c_dataaug.csv', 'result-0989_vote.csv']
    # 0992
    # fileLists = ['removeable/result-0989_vote.csv', 'removeable/result-0920-resnet50.csv', 'result-0979-densenet_4c_aug.csv', 'result-0991_vote.csv']
    # 0993
    fileLists = ['result-0962-inception_resv2_4c_aug.csv', 'result-0992_vote.csv', 'result-0970-inception_4c_aug.csv', 'result-0979-densenet_4c_aug.csv', 'result-0991_vote.csv']

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
    result = true_label == vote_label
    acc = sum(result.astype(np.int64))

    sub = pd.DataFrame({'id': img_id, 'lable': vote_label})
    sub.to_csv("result-{:0>4}_vote.csv".format(acc), index=False, header=False, sep=' ')

def Compare(vote_result,test_truth):
    new_label = pd.read_csv(vote_result, sep=' ', names=['id', 'label'])['label'].values
    true_label = pd.read_csv(test_truth, sep=' ', names=['id', 'label'])['label'].values

    result = true_label == new_label
    acc = sum(result.astype(np.int64)) / len(result)

    print('acc of vote_result',acc)

if __name__ == '__main__':
    Submission_path = "../Submission"

    test_truth = "test_groundtruth.csv"

    MergeResult(Submission_path)
    # Compare(vote_result,test_truth)