#encoding=utf-8
import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

'''
在YOLOv1中，输入图片最终被划分为7x7的gird cell，每个单元格预测2个边界框。YOLOv1最后采用的是全连接层直接对边界框进行预测，其中边界框的宽与高是相对整张图片大小的，而由于各个图片中存在不同尺度和长宽比（scales and ratios）的物体，YOLOv1在训练过程中学习适应不同物体的形状是比较困难的，这也导致YOLOv1在精确定位方面表现较差。YOLOv2则引入了一个anchor boxes的概念，这样做的目的就是得到更高的召回率，yolov1只有98个边界框，yolov2可以达到1000多个（论文中的实现是845个）。还去除了全连接层，保留一定空间结构信息，网络仅由卷积层和池化层构成。

借鉴了RPN网络使用的anchor boxes去预测bounding boxes相对于图片分辨率的offset，通过(x,y,w,h)四个维度去确定anchor boxes的位置

'''
def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors)/num_anchors
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW) 
    ty         = torch.zeros(nB, nA, nH, nW) 
    tw         = torch.zeros(nB, nA, nH, nW) 
    th         = torch.zeros(nB, nA, nH, nW) 
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW) 

    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    for b in xrange(nB):
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
        cur_ious = torch.zeros(nAnchors)
        for t in xrange(50):
            if target[b][t*5+1] == 0:
                break
            gx = target[b][t*5+1]*nW
            gy = target[b][t*5+2]*nH
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        ignore_ix = (cur_ious>sil_thresh).view(nA,nH,nW)   #!!!!!!已更改
        conf_mask[b][ignore_ix] = 0
    if seen < 12800: #只在12800次迭代前加上，目的应该是为了使预测框快速学习到先验框的形状。
       if anchor_step == 4:
           tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
           ty = torch.FloatTensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
       else:
           tx.fill_(0.5)
           ty.fill_(0.5)
       tw.zero_()
       th.zero_()
       coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for b in xrange(nB):
        for t in xrange(50):
            if target[b][t*5+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t*5+1] * nW
            gy = target[b][t*5+2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            gt_box = [0, 0, gw, gh]
            for n in xrange(nA):
                aw = anchors[anchor_step*n]
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if anchor_step == 4:
                    ax = anchors[anchor_step*n+2]
                    ay = anchors[anchor_step*n+3]
                    dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step==4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            #gt_box = [gx, gy, gw, gh]  #更改如下
            gt_box = torch.FloatTensor([gx, gy, gw, gh])
            
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
            #print('[b][best_n][gj][gi]               =====',b,best_n,gj,gi)
            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
            tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
            th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t*5]
            if iou > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1,use_cuda=True):
        super(RegionLoss, self).__init__()
        use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1    #坐标loss的损失权重：1
        self.noobject_scale = 1  #修正系数也就是惩罚系数
        self.object_scale = 5   #无目标及有目标loss为1：5
        self.class_scale = 1    #分类loss的
        self.thresh = 0.6     #IOU阈值
        self.seen = 0
        '''
        在未与true_boxes匹配的anchor_boxes中，与true_boxes的IOU小于0.6的被标记为background，这部分预测正确，未造成损失。但未与true_boxes匹配的anchor_boxes中，若与true_boxes的IOU大于0.6的我们需要计算其损失，因为它未能准确预测background，与true_boxes重合度过高，就是no_objects_loss。而objects_loss则是与true_boxes匹配的anchor_boxes的预测误差。与YOLOv1不同的是修正系数的改变，YOLOv1中no_objects_loss和objects_loss分别是0.5和1，而YOLOv2中则是1和5。

        '''

    def forward(self, output, target):   #载入模型输出与原label计算loss
        #output : (64, 125, 13, 13)
        #注意！！！这里的5*(5+20),外面的5指的是5个bounding box，125指的是5个bounding box的预测值
        #在yolo1中是利用全连接层直接对以上输出值进行回归预测；
        #而在yolo2中引入了聚类产生的anchor box；
        #根据上面的预测输出加入anchor box的尺寸，产生得到真实box的实际位置和大小
        t0 = time.time()
        nB = output.data.size(0)   #batch 64
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)   #H/w=13
        nW = output.data.size(3)
        
        output   = output.view(nB, nA, (5+nC), nH, nW)  #batch:64,anchor:5,(5+20),13,13
        
        #每个box的5个值处理输出：包括规则化sigmiod
        #注意这里的取值方式.index_select(),2表示第二个维度正好对应上面的(5+nc)，后面表示在这个5+nc上取对应维度
        #值，正好对应其坐标x/y及w/h
        #x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB*nA*nH*nW))
        #y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB*nA*nH*nW))
        #w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB*nA*nH*nW)
        #h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB*nA*nH*nW)
        #conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        #置信度
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB*nA*nH*nW))
        
        cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW) #预测box初始化，4代表x\y\w\h
        #grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_x = torch.linspace(0, nW-1, nW).repeat(nB*nA, nH, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        #print(x.data.size(),grid_x.size(),grid_y.size(),y.data.size())
        
        #每个真实box对应的5个值，中心点坐标x，y；及w，h和置信度，结合output输出的bounding box值
        #以及anchor box值，产生真实box的值
        pred_boxes[0] = x.data + grid_x
        pred_boxes[1] = y.data + grid_y
        pred_boxes[2] = torch.exp(w.data) * anchor_w
        pred_boxes[3] = torch.exp(h.data) * anchor_h
        #print('anchor w/h -----------------------:',anchor_w,anchor_h)   #这里的anchor w/h是预先指定的
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
                                                               nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().data[0])

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        #tcls  = Variable(tcls.view(-1)[cls_mask].long().cuda())
        tcls  = Variable(tcls[cls_mask].long().view(-1).cuda())

        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)  

        t3 = time.time()

        #添加几行，使维度对应，没什么实际意义
        coord_mask = coord_mask.view(nB*nA*nH*nW)
        tx=tx.view(nB*nA*nH*nW)
        ty=ty.view(nB*nA*nH*nW)
        tw=tw.view(nB*nA*nH*nW)
        th=th.view(nB*nA*nH*nW)
        tconf = tconf.view(nB*nA*nH*nW)
        conf_mask =conf_mask.view(nB*nA*nH*nW)
             
        '''
        坐标损失：较YOLOv1的改动较大，计算x,y的误差由相对于整个图像（416x416）的offset坐标误差的均方改变为相对于gird cell的offset（这个offset是取sigmoid函数得到的处于（0,1）的值）坐标误差的均方。也将修正系数由5改为了1 。计算w,h的误差由w,h平方根的差的均方误差变为了，w,h与对true_boxes匹配的anchor_boxes的长宽的比值取log函数，和YOLOv1的想法一样，对于相等的误差值，降低对大物体误差的惩罚，加大对小物体误差的惩罚。同时也将修正系数由5改为了1

        '''
        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
        
        
        
        #分类损失：和YOLOv1基本一致，就是经过softmax后，20维向量（数据集中分类种类为20种）的均方误差
        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        
        
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        
        #loss由三部分组成：坐标损失+IOU损失+分类损失
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        #print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
        return loss
