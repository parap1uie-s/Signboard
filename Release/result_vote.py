import pandas as pd
import numpy as np
import os

def MergeResult(Submission_path):

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
    # fileLists = ['result-0962-inception_resv2_4c_aug.csv', 'result-0992_vote.csv', 'result-0970-inception_4c_aug.csv', 'result-0979-densenet_4c_aug.csv', 'result-0991_vote.csv']
    # fileLists = ['result-0961-xception_4c_aug.csv', 'result-0949-nasnet_4c_aug.csv', 'result-0992_vote.csv', 'result-0993_vote.csv']
    # 0994
    # fileLists = ['result-0961-xception_4c_aug.csv', 'result-0935-CRNN.csv', 'result-0992_vote.csv', 'result-0993_vote.csv', 'result-0979-densenet_4c_aug.csv']
    # 0995
    fileLists = ['result-0988-densenet_4c_aug.csv', 'result-0935-CRNN.csv', 'result-0992_vote.csv', 'result-0962-nasnet_4c_aug.csv', 'result-0994_vote.csv']

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
    sub.to_csv("result_vote.csv".format(acc), index=False, header=False, sep=' ')

if __name__ == '__main__':
    Submission_path = "Submission/"
    MergeResult(Submission_path)