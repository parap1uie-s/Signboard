import os
import numpy as np
import pandas as pd
from dataset.dataset import testdataset, collate_fn
import torch
from torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from math import ceil
from  torch.nn.functional import softmax
from models.multiscale_resnet import multiscale_resnet
test_transforms= transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

mode ="test"

datapath = '/home/Signboard/datasets'
train_pd = pd.read_csv(os.path.join(datapath, "train_split.txt"), sep=' ', names=['filepath', 'classid'],dtype={"filepath":"str", "classid":"str"})
val_pd = pd.read_csv(os.path.join(datapath, "val_split.txt"), sep=' ', names=['filepath', 'classid'],dtype={"filepath":"str", "classid":"str"})

true_test_pb = pd.read_csv(os.path.join(datapath, "test.txt"), sep=' ', names=['filepath'],dtype={"filepath":"str"})
true_test_pb['classid'] =1

test_pd = true_test_pb if mode=="test" else val_pd
print(test_pd.head())

data_set = {}
data_set['test'] = testdataset(imgroot=os.path.join(datapath, "test"), anno_pd=test_pd,
                             transforms=test_transforms,
                             )
data_loader = {}
data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=8, num_workers=4,
                                           shuffle=False, pin_memory=True, collate_fn=collate_fn)

model_name = 'resnet50-out'
resume = 'resnet50/weights-10-230-[0.9829].pth'

# model =resnet50(pretrained=True)
# model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
# model.fc = torch.nn.Linear(model.fc.in_features,100)
model =multiscale_resnet(num_class=100)

print('resuming finetune from %s'%resume)
model.load_state_dict(torch.load(resume))
model = model.cuda()
model.eval()
criterion = CrossEntropyLoss()

test_size = ceil(len(data_set['test']) / data_loader['test'].batch_size)
test_preds = np.zeros((len(data_set['test'])), dtype=np.float32)
true_label = np.zeros((len(data_set['test'])), dtype=np.int)
idx = 0
test_loss = 0
test_corrects = 0
for batch_cnt_test, data_test in enumerate(data_loader['test']):
    # print data
    print("{0}/{1}".format(batch_cnt_test, int(test_size)))
    inputs, labels = data_test
    inputs = Variable(inputs.cuda())
    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
    # forward
    outputs = model(inputs)

    # statistics
    if isinstance(outputs, list):
        loss = criterion(outputs[0], labels)
        loss += criterion(outputs[1], labels)
        temp = 0
        for output in outputs:
            temp += output
        outputs = temp / len(outputs)
    else:
        loss = criterion(outputs, labels)
    _, preds = torch.max(outputs, 1)

    test_loss += loss.data[0]
    batch_corrects = torch.sum((preds == labels)).data[0]
    test_corrects += batch_corrects
    test_preds[idx:(idx + labels.size(0))] = preds
    true_label[idx:(idx + labels.size(0))] = labels.data.cpu().numpy()
    # statistics
    idx += labels.size(0)
test_loss = test_loss / test_size
test_acc = 1.0 * test_corrects / len(data_set['test'])
print('test-loss: %.4f ||test-acc@1: %.4f'
      % (test_loss, test_acc))

test_pred = test_pd[['filepath']].copy()
test_pred['classid'] = list(test_preds)
test_pred['classid'] = test_pred['classid'].apply(lambda x: int(x)+1)
test_pred[['filepath',"classid"]].to_csv('result.csv'.format(model_name,mode) ,sep=" ", header=None, index=False)
print(test_pred.info())