import pandas as pd
import numpy as np
import os
import itertools
from multiprocessing import Process

def pro_do(fileLists, true_label, labels, ind):
    bestAcc = 0
    bestComb = []
    for fileList in itertools.combinations(range(0,len(fileLists)), ind):
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
        if acc > bestAcc:
            bestAcc = acc
            bestComb = fileList

    print("Comb:{}".format(i))
    print(bestAcc)
    print([fileLists[i] for i in bestComb])

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

    processes = []
    for i in range(2, len(fileLists) + 1):
        p = Process(target=pro_do, args=(fileLists, true_label, labels, i))
        p.daemon = True    #加入daemon
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    Submission_path = "../Submission"

    test_truth = "test_groundtruth.csv"

    GridMergeResult(Submission_path)
    # Compare(vote_result,test_truth)