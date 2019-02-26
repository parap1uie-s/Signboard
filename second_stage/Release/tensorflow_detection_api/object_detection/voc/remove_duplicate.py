

target_path = "VOCdevkit/VOC2012/ImageSets/"
datasets = ["datasets/train.txt","datasets/val.txt"]
for file in datasets:
    with open(file,'r') as f:
        unique = set()
        line = f.readline()
        while line:
            line = line.split('.')[0]
            unique.add(line)
            print(line)
            line = f.readline()



        with open(target_path+'/'+file.split('/')[1],'w') as f1:
            for line in unique:
                f1.write(line+'\n')
