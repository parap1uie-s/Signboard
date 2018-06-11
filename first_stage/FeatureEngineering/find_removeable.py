import pandas as pd
import numpy as np
import os
import itertools

def MergeResult(Submission_path):
    fileLists = os.listdir(Submission_path)

    true_label = pd.read_csv("test_groundtruth.csv", sep=' ', names=['id', 'label_true'])

    for fileList in itertools.combinations(fileLists, 2):
        if fileList[0].split(".")[-1] != "csv" or fileList[1].split(".")[-1] != "csv":
            continue
        # 统计每一行次数出现最多的数字
        result_handle_1 = pd.read_csv(os.path.join(Submission_path,fileList[0]),sep=' ',names=['id','label_1'])
        result_handle_2 = pd.read_csv(os.path.join(Submission_path,fileList[1]),sep=' ',names=['id','label_2'])
        handle_csv = true_label.merge(result_handle_1, how='left', on='id').merge(result_handle_2, how="left", on="id")

        result_1_index = set(handle_csv[ handle_csv['label_1'] == handle_csv['label_true'] ].index)
        result_2_index = set(handle_csv[ handle_csv['label_2'] == handle_csv['label_true'] ].index)
        if result_1_index & result_2_index == result_2_index:
            print("Dropable:{}, compared to {}".format(fileList[1], fileList[0]))
        elif result_1_index & result_2_index == result_1_index:
            print("Dropable:{}, compared to {}".format(fileList[0], fileList[1]))

if __name__ == '__main__':
    Submission_path = "../Submission"

    test_truth = "test_groundtruth.csv"

    MergeResult(Submission_path)
    # Compare(vote_result,test_truth)