from __future__ import print_function
import sys
if len(sys.argv) < 4:
    print('Usage:')
    print('python eval.py datacfg cfgfile weight1 weight2 ...')
    exit()

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
import codecs

# Training settings
datacfg       = sys.argv[1]
cfgfile       = sys.argv[2]

data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]

#voc  

#trainlist     = data_options['ori_valid']
#trainlist     = data_options['9.18_3_600_adv_test']
#trainlist     = data_options['9.18_19_1500_adv_test']
#trainlist     = data_options['9.18_14_1500_adv_test_gray_repla1']
trainlist     = data_options['9.18_14_1500_adv_test_gray_repla2']
labelpath     = data_options['voc_lab_path']

#trainlist     = data_options['10.14_oriimg_detec']
#labelpath     = data_options['voc_lab_path']

#inria , cfg/inria.data
#trainlist     = data_options['train3'] #inpa
#trainlist     = data_options['train2'] #rep1
#trainlist     = data_options['train1'] #adv
#trainlist     = data_options['train'] #ori
#labelpath     = data_options['lab_path']


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
    #分别计数各个类别得total\proposals\correct
    total       = [0 for i in range(num_classes)]
    proposals   = [0 for j in range(num_classes)]
    correct     = [0 for k in range(num_classes)]
    
    #14_1500

    #darknet
    #result_file=codecs.open('./9.19_darknet_ori.txt','w')
    #result_file=codecs.open('./9.19_darknet_3_1500_adv_test.txt','w') 
    #result_file=codecs.open('./9.19_darknet_3_600_adv_test_rep.txt','w')
    #result_file=codecs.open('./9.19_darknet_selftrain190_ori.txt','w')
    #result_file=codecs.open('./9.19_darknet_selftrain190_3_600_adv_test.txt','w')
    #result_file=codecs.open('./9.19_darknet_selftrain190_3_600_adv_test_rep.txt','w')
    #tiny
    #result_file=codecs.open('./9.19_tiny_ori.txt','w')                              
    #result_file=codecs.open('./9.19_tiny_3_600_adv_test.txt','w')                 
    #result_file=codecs.open('./9.19_tiny_3_600_adv_test_rep.txt','w')             
    #result_file=codecs.open('./9.19_tiny_selftrain190_ori.txt','w')                 
    #result_file=codecs.open('./9.19_tiny_selftrain190_3_1500_adv_test.txt','w')    
    #result_file=codecs.open('./9.19_tiny_selftrain190_14_1500_adv_test_rep.txt','w')

    ##entroy_gray_replace
    #result_file=codecs.open('./inria_inpaint_dark.txt','w')
    #result_file=codecs.open('./9.18_14_1500_adv_test_gray_repla2_pred.txt','w')
    #result_file=codecs.open('./10.14_oriimg_tiny_ori_detec.txt','w')

    #inria entro_gray_inpai
    #result_file=codecs.open('./inria_ori_dark1.txt','w')
    
    #result_file=codecs.open('./12.20_dark_9_1500_adv.txt','w')
    #result_file=codecs.open('./12.20_tiny_19_1500_adv.txt','w')

    #result_file=codecs.open('./12.23_ent_repla_tiny.txt','w')
    #result_file=codecs.open('./12.23_inria_ent_tiny.txt','w')
    #result_file=codecs.open('./12.23_ori_our_dark.txt','w')
    result_file=codecs.open('./12.23_test.txt','w')

    for batch_idx, (data, target) in enumerate(test_loader):
        t1=time.time()
        if batch_idx%10==0:
            print('--------------------batch_idx-------------------',batch_idx)
        #print('========total================',total,proposals,correct)
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        t2=time.time()
        output = model(data).data
        t3=time.time()
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        #print('-----------all_boxes--------',type(all_boxes),len(all_boxes[0]),len(all_boxes[1]),len(all_boxes[2]),all_boxes[0])
        #print('----------target---------',type(target),target.shape,target[0,:])
        t4=time.time()
        #all_boxes==[box=[bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]],包含所有预测到的box
        #target,50*5,c\x\y\w\h
        
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            #print('boxes',len(boxes))
            boxes = nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)  #真实lables中有多少个box=[c\x\y\w\h]
            print('boxes',len(boxes),num_gts)    
            for kk in range(num_gts):
                #print('=========truths[kk][0]==========',truths[kk][0].item())
                tclass=int(truths[kk][0].item())
                total[tclass] = total[tclass] + 1 #计数对应类别个数
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
                    #print('-------------iou-------',iou)
                    if iou!=0 and iou.item() > best_iou:  #tensor转数字加.item()
                        best_j = j
                        best_iou = iou.item()   
                #print('-------best_iou-----',best_iou,iou_thresh,(boxes[best_j][6]).float(),(box_gt[6]).float())
                #if best_iou > iou_thresh and (boxes[best_j][6]).float().item()-14.0 == (box_gt[6]).float().item(): #注意这里的-14是使前后类别对应
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
    print('map--p  r  f--:'+str(round(precision2/num_classes,3))+'\t'+str(round(recall2/num_classes,3))+'\t'+str(round(fscore2/num_classes,3)))
for i in range(3, len(sys.argv)):
    weightfile = sys.argv[i]
    model.module.load_weights2(weightfile)
    logging('evaluating ... %s' % (weightfile))
    test()
