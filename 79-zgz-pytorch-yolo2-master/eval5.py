#encoding=utf-8
from __future__ import print_function
import sys
import time
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable

import dataset3 as dataset
import random
import math
from utils import *
from cfg import parse_cfg
from darknet import Darknet
import codecs

# Training settings
datacfg='/home/zgz/adversarial_samples/adversarial-yolo-master2/cfg/my_voc.data'
#cfgfile='/home/zgz/pytorch-yolo2-master/cfg/yolo-voc.cfg'
cfgfile='./cfg/yolo-voc.cfg'
weightfile='/home/zgz/pytorch-yolo2-master/9.18_my_models_detect/000021.weights'
#weightfile='/home/zgz/pytorch-yolo2-master/yolov2-voc.weights'

data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]

trainlist     = data_options['train']
labelpath     = data_options['voc_lab_path']
trainlist     = data_options['9.16_NDSS_TrojanNN_test']
labelpath     = data_options['9.16_NDSS_TrojanNN_test_label']
trainlist     = data_options['9.16_LaVAN_test1']      
labelpath     = data_options['9.16_LaVAN_test_label1']
#trainlist     = data_options['9.16_LaVAN_test']      
#labelpath     = data_options['9.16_LaVAN_test_label']

gpus          = data_options['gpus']  # e.g. 0,1,2,3
num_workers   = int(data_options['num_workers'])

batch_size    = int(net_options['batch'])

#Train parameters
use_cuda      = True
seed          = 22222
eps           = 1e-5

# Test parameters
#conf_thresh   = 0.25
#nms_thresh    = 0.4
conf_thresh   = 0.25
nms_thresh    = 0.4

iou_thresh    = 0.5
# conf_thresh   = 0.0
# nms_thresh    = 0.2
# iou_thresh    = 0.2

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
    dataset.MyDataset(trainlist, labelpath,shape=(init_width, init_height),
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]), train=False),
    batch_size=batch_size, shuffle=False, **kwargs)

if use_cuda:
    model = torch.nn.DataParallel(model).cuda()

def test():
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    model.eval()
    print('model---------',model)
    num_classes = model.module.num_classes
    anchors     = model.module.anchors
    num_anchors = model.module.num_anchors
    #???????????????????????????total\proposals\correct
    total       = [0 for i in range(num_classes)]
    proposals   = [0 for j in range(num_classes)]
    correct     = [0 for k in range(num_classes)]
    
    #result_file=codecs.open('./darknet_000190_weights_result_adv.txt','w')
    #result_file=codecs.open('./darknet_000190_weights_result.txt','w')
    #result_file=codecs.open('./darknet544_000190_weights_result_adv.txt','w')
    #result_file=codecs.open('./darknet544_000020_weights_result.txt','w')
    #result_file=codecs.open('./yolo_weights_result_adv.txt','w')
    #result_file=codecs.open('./yolo_weights_result.txt','w')
    #result_file=codecs.open('./yolov2-416_result_adv_obj5.txt','w')
    #result_file=codecs.open('./14_901_tinycoco.txt','w')
    #result_file=codecs.open('./14_900_tinyvoc.txt','w')
    result_file=codecs.open('./9.29_LaVan_test1_pred.txt','w')
    #result_file=codecs.open('./9.22_NDSS_test_pred','w')

    #result_file=codecs.open('./yolov2-tiny-voc_train__result2_adv.txt','w')
    #result_file=codecs.open('./yolo_weights_result.txt','w') 

    for batch_idx, (data, target) in enumerate(test_loader):
        t1=time.time()
        if batch_idx%20==0:
            print('--------------------batch_idx-------------------',batch_idx)
        #print('========total================',total,proposals,correct)
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        t2=time.time()
        output = model(data).data
        t3=time.time()
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
    #    print('-----------all_boxes--------',type(all_boxes),len(all_boxes[0]),len(all_boxes[1]),len(all_boxes[2]),all_boxes[0])
     #   print('----------target---------',type(target),target.shape,target[0,:])
        t4=time.time()
        #all_boxes==[box=[bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]],????????????????????????box
        #target,50*5,c\x\y\w\h
        
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            #print('boxes',len(boxes))
            boxes = nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)  #??????lables???????????????box=[c\x\y\w\h]
            #print('num_gts,boxes',num_gts,boxes)
            
            for kk in range(num_gts):
                #print('=========truths[kk][0]==========',truths[kk][0].item())
                tclass=int(truths[kk][0].item())
                total[tclass] = total[tclass] + 1 #????????????????????????
                #print('+++++++++++++total[tclass]+++++++++++++++',total[tclass])
    
            for i in range(len(boxes)):
                if boxes[i][4] > conf_thresh:
                    proposals[boxes[i][6]] = proposals[boxes[i][6]]+1
            
            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
            #        print('-------------iou-------',iou)
                    if iou!=0 and iou.item() > best_iou:  #tensor????????????.item()
                        best_j = j
                        best_iou = iou.item()   
                #print('-------best_iou-----',best_iou,iou_thresh,(boxes[best_j][6]).float(),(box_gt[6]).float())
                #if best_iou > iou_thresh and (boxes[best_j][6]).float().item()-14.0 == (box_gt[6]).float().item(): #???????????????-14????????????????????????
                #print('---------------------------',boxes[best_j][6].float().item())
                if best_iou > iou_thresh and (boxes[best_j][6]).float().item() == (box_gt[6]).float().item():
                    tclass=int(truths[i][0].item())
                    correct[tclass] = correct[tclass]+1
        t5=time.time()
        #print('-----------time---------------',t2-t1,t3-t2,t4-t3,t5-t4)
    precision2,recall2,fscore2=0,0,0
    result_file.write('\t'+'precision:'+'\t'+'recall:'+'\t'+'fscore:\n')
    for i in range(num_classes):
        precision = 1.0*correct[i]/(proposals[i]+eps)
        recall = 1.0*correct[i]/(total[i]+eps)
        fscore = 2.0*precision*recall/(precision+recall+eps)
        precision2,recall2,fscore2=precision2+precision,recall2+recall,fscore2+fscore
        logging("precision    %d   : %f, recall: %f, fscore: %f" % (i,precision, recall, fscore))
        result_file.write(str(i)+'\t'+str(round(precision,3))+'\t'+str(round(recall,3))+'\t'+str(round(fscore,3))+'\n')
    result_file.write('map--p  r  f--: '+str(round(precision2/num_classes,3))+'\t'+str(round(recall2/num_classes,3))+'\t'+str(round(fscore2/num_classes,3)))
    print('----------result_file-------------',result_file)
    
    
model.module.load_weights(weightfile)
logging('evaluating ... %s' % (weightfile))
test()
