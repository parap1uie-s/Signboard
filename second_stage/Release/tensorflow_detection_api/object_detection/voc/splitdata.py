
import os
datapath = "dataset\\categories"
train_set = "train.txt"
val_set = "val.txt"

ftrain = open(train_set,'w')
fval = open(val_set,'w')
categories = os.listdir(datapath)
for cate in categories:
    cate_path = os.path.join(datapath,cate)
    if os.path.isdir(cate_path):
        files = os.listdir(cate_path)
        fn = 0
        # 统计文件个数
        for file in files:
            file_path = os.path.join(cate_path,file)
            if os.path.isfile(file_path):
                fn += 1
        train_num = int(fn*0.9)
        val_num = fn-train_num

        for train_index in range(0,train_num,1):
            ftrain.write(files[train_index].split('.')[0]+' '+cate+'\n')
            # ftrain.write(files[train_index].split('.')[0] + '\n')
        for val_index in range(train_num,fn,1):
            fval.write(files[val_index].split('.')[0]+' '+cate+'\n')
            # fval.write(files[val_index].split('.')[0] + '\n')

ftrain.close()
fval.close()