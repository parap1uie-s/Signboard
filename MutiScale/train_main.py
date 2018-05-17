#coding=utf-8
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.dataset import collate_fn, dataset
import torch
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.train_util import train, trainlog
from  torch.nn import CrossEntropyLoss
import logging
from models.multiscale_resnet import multiscale_resnet
datapath = '/home/Signboard/datasets'
train_pd = pd.read_csv(os.path.join(datapath, "train_split.txt"), sep=' ', names=['filepath', 'classid'],dtype={"filepath":"str", "classid":"str"})
val_pd = pd.read_csv(os.path.join(datapath, "val_split.txt"), sep=' ', names=['filepath', 'classid'],dtype={"filepath":"str", "classid":"str"})

'''数据扩增'''
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(degrees=20),
        transforms.RandomResizedCrop(224,scale=(0.49,1.0)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),   # 0-255 to 0-1
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

save_dir = 'resnet50'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = '%s/trainlog.log'%save_dir
trainlog(logfile)
data_set = {}
data_set['train'] = dataset(imgroot=datapath+"/train/",anno_pd=train_pd,
                           transforms=data_transforms["train"],
                           )
data_set['val'] = dataset(imgroot=datapath+"/val/",anno_pd=val_pd,
                           transforms=data_transforms["val"],
                           )
dataloader = {}
dataloader['train']=torch.utils.data.DataLoader(data_set['train'], batch_size=4,
                                               shuffle=True, num_workers=4,collate_fn=collate_fn)
dataloader['val']=torch.utils.data.DataLoader(data_set['val'], batch_size=4,
                                               shuffle=True, num_workers=4,collate_fn=collate_fn)
'''model'''
# model =resnet50(pretrained=True)
# model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
# model.fc = torch.nn.Linear(model.fc.in_features,100)
model =multiscale_resnet(num_class=100)
base_lr =0.001
resume = 'resnet50/weights-9-47-[0.9780].pth'
# resume = None
if resume:
    logging.info('resuming finetune from %s'%resume)
    model.load_state_dict(torch.load(resume))
model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
criterion = CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
best_acc,best_model_wts = train(model,
                                  epoch_num=50,
                                  start_epoch=0,
                                  optimizer=optimizer,
                                  criterion=criterion,
                                  exp_lr_scheduler=exp_lr_scheduler,
                                  data_set=data_set,
                                  data_loader=dataloader,
                                  save_dir=save_dir,
                                  print_inter=50,
                                  val_inter=400,
                                  )