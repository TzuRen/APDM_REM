from __future__ import print_function
import sys
import time
import torch
import torch.nn as nn
from torchvision import transforms
import dataset_test as datasets
import os

trainlist     = '/home/zgz/pytorch-yolo2-master/voc_train.txt'
gpus          = '0,1,2,3'
num_workers   = 2
batch_size    = 64
use_cuda     = True
seed        = int(time.time())

torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

init_width        = 416 #model.width
init_height       = 416 #model.height
print('--------------------init_width,h------------',init_width,init_height)
kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

def train(epoch_len):
    t0 = time.time()
    train_loader = torch.utils.data.DataLoader(
        datasets.listDataset(trainlist, shape=(init_width,init_height),
                       shuffle=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]), 
                       train=True, 
                       seen=0,
                       batch_size=batch_size,
                       num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)
    numb=0
    t1 = time.time()
    print('t1-t0 :',t1-t0)
    for epoch in range(1,epoch_len):
        
        t2 = time.time()
        print('t2-t1 :',t2-t1)
        for batch_idx, (data, target) in enumerate(train_loader):
            t3 = time.time()
            print('t3-t2 , batch_idx :',t3-t2,batch_idx)
            pass

        t4 = time.time()
    t5 = time.time()

train(10)