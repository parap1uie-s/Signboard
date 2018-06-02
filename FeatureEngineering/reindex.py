import pandas as pd
import numpy as np
import os

def equal(a, b):
    return int(a) == int(b)

if __name__ == '__main__':
    Submission_path = "../FeatureEngineering"
    test_truth = "test_groundtruth.csv"

    gt_result = pd.read_csv(test_truth , sep=' ', names=['filepath', 'classid_true'],
        dtype={"filepath":"str", "classid_true":"int"})

    fileLists = os.listdir(Submission_path)
    for file in fileLists:
        if file.split(".")[-1] != "csv":
            continue
        if file == test_truth:
            continue
        result_handle = pd.read_csv(os.path.join(Submission_path,file),sep=' ',names=['filepath','classid_pred'], 
            dtype={"filepath":"str", "classid_pred":"int"})
        compare = gt_result.merge(result_handle, how='left', on='filepath')
        compare.loc[:,["filepath","classid_pred"]].to_csv(file, index=False, header=False, sep=' ')

        # check
        compare['res'] = compare.apply(lambda row: equal(row['classid_true'], row['classid_pred']), axis=1)

        acc = compare['res'].sum()
        print("{} - {}".format(file, acc))