#encoding=utf-8
#性能对比评估：
#评估图片地址：/home/zgz/adversarial_samples/adversarial-yolo-master/inria/Train/pos
#其真实label：/home/zgz/adversarial_samples/adversarial-yolo-master/inria/Train/pos/yolo-labels

#注意这里的labels信息：第一列为类别0代表人，后面四个值分别表示
#中心归一化后横坐标/纵坐标、归一化宽度/高度

#这里需要对比的几种情况： 根提供的labels进行对比
#1.原始图像检测效果；2.带了对抗块之后的检测效果；3.对抗块初步检测替换(简单灰度替换)之后的检测效果
#4.图像块检测恢复之后的检测效果

from __future__ import print_function
import sys
import time
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable

import dataset
import random
import math
from utils import *
from cfg import parse_cfg
from darknet import Darknet

# Training settings
datacfg       = './cfg/inria.data'
cfgfile       =  './cfg/yolo-voc.cfg'
weightfile = './yolov2.weights'

print('evaluating ... %s' % (weightfile))

data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]

trainlist     = data_options['train']  #注意对于这里的数据集来说，使用train作为test，而且路径有变需要更改listDataset
testlist      = data_options['test']   #这里的test是没有labels
gpus          = data_options['gpus']  # e.g. 0,1,2,3
num_workers   = int(data_options['num_workers'])

batch_size    = int(net_options['batch'])

#Train parameters
use_cuda      = True
seed          = 22222
eps           = 1e-5

# Test parameters
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

###############
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

model       = Darknet(cfgfile)
model.print_network()

init_width        = model.width
init_height       = model.height

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
    dataset.MyDataset(trainlist, shape=(init_width, init_height),
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]), train=False),
    batch_size=batch_size, shuffle=False, **kwargs)

if use_cuda:
    model = torch.nn.DataParallel(model).cuda()

model.module.load_weights(weightfile)
    
def test():
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    model.eval()
    num_classes = model.module.num_classes
    anchors     = model.module.anchors
    num_anchors = model.module.num_anchors
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data).data
        print('output size : ',output.shape)
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        print('all_boxes size : ',all_boxes.shape)
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
                print(type(best_iou),type(iou_thresh))
                if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                    correct = correct+1

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    print("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))

#for i in range(3, len(sys.argv)):

test()
