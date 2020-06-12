#encoding=utf-8
#使用一批自己生成的含有对抗块(0-5个)的对抗图像,指定所有的对抗块的类标为stop sign，
#在voc20类当中是没有这个类的，直接用这个类stop sign替换原voc的最后一个类标，
#指定所有对抗块的label=19，及其坐标，进行训练yolo2，并保存最后训练权重
from __future__ import print_function
import sys
'''
if len(sys.argv) != 4:
    print('Usage:')
    print('python train.py datacfg cfgfile weightfile')
    exit()
'''
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
import dataset as datasets
from torch.autograd import Variable

import random
import math
import os
from utils import *
from cfg import parse_cfg
from region_loss import RegionLoss
from darknet import Darknet
from models.tiny_yolo import TinyYoloNet


# Training settings
#datacfg       = sys.argv[1]
#cfgfile       = sys.argv[2]
#weightfile    = sys.argv[3]

datacfg='cfg/helmep.data'  #相应做了一系列修改，
cfgfile='cfg/yolo-voc.cfg'
#weightfile='yolov2-voc.weights'
weightfile='helmep_models/000363.weights'
#cfgfile='cfg/tiny-yolo-voc.cfg'
#weightfile='yolov2-tiny-voc.weights'
#weightfile='./my_models_repair_dark/000190.weights'
#weightfile='./my_models_repair_dark544_32/000030.weights'
#weightfile='./my_models_repair/000030.weights'

data_options  = read_data_cfg(datacfg)   #以字典形式返回配置
net_options   = parse_cfg(cfgfile)[0]    #以list形式返回所有block，这里只取第一个block即超参数配置

trainlist     = data_options['train']
#testlist      = data_options['valid']
backupdir     = data_options['backup']
nsamples      = file_lines(trainlist)
gpus          = data_options['gpus']  # e.g. 0,1,2,3
ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])
print (backupdir)

batch_size    = int(net_options['batch'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]

#Train parameters
max_epochs    = max_batches*batch_size/nsamples+1
use_cuda      = True
seed          = int(time.time())
eps           = 1e-5
save_interval = 1  # epoches
dot_interval  = 60  # batches

# Test parameters
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

if not os.path.exists(backupdir):
    os.mkdir(backupdir)
    
###############
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

#定义模型
model       = Darknet(cfgfile)
region_loss = model.loss
model.load_weights(weightfile)
model.print_network()
#model.seen=0
region_loss.seen  = model.seen
processed_batches = model.seen/batch_size

init_width        = model.width
init_height       = model.height
init_epoch        = model.seen/nsamples 
print('--------------------init_width,h------------',init_width,init_height)
kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
'''
test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(testlist, shape=(init_width, init_height),
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]), train=False),
    batch_size=batch_size, shuffle=False, **kwargs)
'''
if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

params_dict = dict(model.named_parameters())
params = []
for key, value in params_dict.items():
    if key.find('.bn') >= 0 or key.find('.bias') >= 0:
        params += [{'params': [value], 'weight_decay': 0.0}]
    else:
        params += [{'params': [value], 'weight_decay': decay*batch_size}]
optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)

def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def train(epoch):
    global processed_batches
    
    t0 = time.time()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    #cur_model.seen=0
    train_loader = torch.utils.data.DataLoader(
        datasets.listDataset(trainlist, shape=(init_width,init_height),
                       shuffle=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]), 
                       train=True, 
                       seen=cur_model.seen,
                       batch_size=batch_size,
                       num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)

    number=0
    for epoch in range(1,500):

        #lr = adjust_learning_rate(optimizer, processed_batches)
        #print('-----------processed_batches---------',processed_batches)
        lr = adjust_learning_rate(optimizer, number)
        logging('epoch %d, lr %f' % (epoch, lr))
        model.train()
        t1 = time.time()
        avg_time = torch.zeros(9)
        loss=0
        #print('++++++++++++++++++++++++',len(train_loader.dataset))
        for batch_idx, (data, target) in enumerate(train_loader):
            
            #print('--------------------data---------------',type(data),data.shape) #<class 'torch.Tensor'> (64, 3, :, :)
            
            #for kk in range(64):
            #im = transforms.ToPILImage('RGB')(data[1,:,:,:])
            #im.save('./sele_adv_repla/'+str(epoch)+'_.png')
            
            adjust_learning_rate(optimizer, number)
            processed_batches = processed_batches + 1
            number=number+1
            #if (batch_idx+1) % dot_interval == 0:
            #    sys.stdout.write('.')

            if use_cuda:
                data = data.cuda()
                #target= target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            #数据加载到模型
            output = model(data)    #size:(64, 125, 13, 13),一个batch 64张图片，每张图片13*13*5*(5+20)
            #print('ouput size -------------:',output.size())
            region_loss.seen = region_loss.seen + data.size(0)

            #计算batch个样本的预测与标签loss值
            loss = region_loss(output, target)

            loss.backward()
            optimizer.step()
            
            
            if batch_idx%30==0:
                print('----batch_idx---loss--:',batch_idx,loss.item())
            
        logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
        
        if (epoch+2) % 11 == 0:
            logging('save weights to %s/%06d.weights' % (backupdir, epoch+2))
            cur_model.seen = (epoch + 2) * len(train_loader.dataset)
            cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+2))

def test(epoch):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    model.eval()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    num_classes = cur_model.num_classes
    anchors     = cur_model.anchors
    num_anchors = cur_model.num_anchors
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data).data
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)
     
            total = total + num_gts
    
            for i in range(len(boxes)):
                if boxes[i][4] > conf_thresh:
                    proposals = proposals+1

            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                    correct = correct+1

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))

evaluate = False
if evaluate:
    logging('evaluating ...')
    test(0)
else:
    train(max_epochs-init_epoch)
    #test(epoch)
