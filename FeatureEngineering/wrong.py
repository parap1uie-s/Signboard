import pandas as pd
import numpy as np
import os

def equal(a, b):
    return int(a) == int(b)

if __name__ == '__main__':
    Submission_path = "../Submission"
    test_truth = "test_groundtruth.csv"

    gt_result = pd.read_csv(test_truth , sep=' ', names=['filepath', 'classid'],
        dtype={"filepath":"str", "classid":"int"})
    wrong_result = [False] * 1000
    fileLists = os.listdir(Submission_path)
    for file in fileLists:
        if file.split(".")[-1] != "csv":
            continue
        result_handle = pd.read_csv(os.path.join(Submission_path,file),sep=' ',names=['filepath','classid'], 
            dtype={"filepath":"str", "classid":"int"})
        compare = gt_result.merge(result_handle, how='left', on='filepath')
        compare = compare.apply(lambda row: equal(row['classid_x'], row['classid_y']), axis=1).tolist()
        wrong_result = [ (compare[k] or res) for k,res in enumerate(wrong_result)]

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

    label_fre = []
    for value in labels:
        beitai = np.unique(value)
        label_fre.append( beitai )
    sub = pd.DataFrame({'filepath': img_id, 'freq': label_fre})
    gt_result['res'] = wrong_result
    gt_result = gt_result.merge(sub, how='left', on='filepath')
    # gt_result[gt_result['res'] == False].loc[:,['filepath','classid','freq']].to_csv("wrong.csv", index=False, header=True, sep=',')
    print( gt_result[gt_result['res'] == False].loc[:,['filepath','classid','freq']] )

        