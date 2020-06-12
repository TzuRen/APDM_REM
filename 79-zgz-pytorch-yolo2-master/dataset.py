#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import read_truths_args, read_truths
from image import *
import codecs
import cv2
import math
from scipy import misc

class listDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

       
        if self.train and index % 64== 0:
            
            if self.seen < 4000*64:
               width = 13*32
               self.shape = (width, width)
            elif self.seen < 8000*64:
               width = (random.randint(0,3) + 13)*32
               self.shape = (width, width)
            elif self.seen < 12000*64:
               width = (random.randint(0,5) + 12)*32
               self.shape = (width, width)
            elif self.seen < 16000*64:
               width = (random.randint(0,7) + 11)*32
               self.shape = (width, width)
            else: # self.seen < 20000*64:
               width = (random.randint(0,9) + 10)*32
               self.shape = (width, width)

        if not self.train:
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
        else:
            #print('==================',)
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
    
            #labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            labpath=imgpath.replace('/img/', '/txt/').replace('.jpg', '.txt')
            label = torch.zeros(50*5)
            #if os.path.getsize(labpath):
            #tmp = torch.from_numpy(np.loadtxt(labpath))
            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
            except Exception:
                tmp = torch.zeros(1,5)
            #tmp = torch.from_numpy(read_truths(labpath))
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            #print('labpath = %s , tsz = %d' % (labpath, tsz))
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return (img, label)

#这里同样加载的为voc的原图像数据包括lables，但是需要做一些更改
#就是说在目标范围内进行随机块值的替换、然后修复，然后再送入网络进行训练一个更鲁棒的yolo网络
#1、目标范围内随机块替换
#2、opencv根据替换后的图像及其mask进行修复
#最后返回修复后的图像
class listDataset2(Dataset):  

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        #print('-------------s-een---',self.seen)
        if self.train and index % 64== 0:
            if self.seen < 4000*64:
               width = 13*32
               self.shape = (width, width)
            elif self.seen < 8000*64:
               width = (random.randint(0,3) + 13)*32
               self.shape = (width, width)
            elif self.seen < 12000*64:
               width = (random.randint(0,5) + 12)*32
               self.shape = (width, width)
            elif self.seen < 16000*64:
               width = (random.randint(0,7) + 11)*32
               self.shape = (width, width)
            else: # self.seen < 20000*64:
               width = (random.randint(0,9) + 10)*32
               self.shape = (width, width)

        if not self.train:
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
        else:
#             print('===============imgpath==================',index,self.shape)  #连续执行一个batch的img！！！！！！！！！！！1
            img = Image.open(imgpath).convert('RGB')
            
            if self.shape:
                img = img.resize(self.shape)
    
            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            #print('===============labpath==================',labpath)
            #1、随机块替换,输入：原图像、及label路径；输出替换后图像及对应mask
            #print('====================================================',img.size[0],img.size[1])
            
            img_rep=rand_repl(img,labpath)
            img=Image.fromarray(img_rep.astype('uint8')).convert('RGB')
            #这里需要注意的是上面是Image操作的，而修复需要转为opencv的
        
            #2、块修复,输入如上输出，返回修复后图像
            #img2 = cv2.imread(imgpath)
            #img=patch_repa(img_rep,img_mask)
            
            #原label不变
            
            label = torch.zeros(50*5)
            #if os.path.getsize(labpath):
            #tmp = torch.from_numpy(np.loadtxt(labpath))
            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
            except Exception:
                tmp = torch.zeros(1,5)
            #tmp = torch.from_numpy(read_truths(labpath))
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            #print('labpath = %s , tsz = %d' % (labpath, tsz))
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return (img, label)

#随机块替换,随机块大小指定为占目标大小的的1/6-1/2，在范围内随机取位置/大小
#对抗块替换：/home/zgz/adversarial_samples/adversarial-yolo-master/gen_adv_voc3/100_3.png
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])   分别*255获取三通道各像素均值
def rand_repl(img,labpath):
    
    img2=np.array(img)
    img_w=img.size[0]
    img_h=img.size[1]
    
    lines=codecs.open(labpath,'r').readlines()
    for line in lines:
        # random image erasing
        rand=random.randint(0,10)
        if rand<3:
            continue
		# bounding box
        box=line.split(' ')[1:]
        box_w=float(box[2])
        box_h=float(box[3])
		#random number
		num=random.randint(0,3)
		for i in range(num):
			# random size
			adv_size_w=random.uniform(0.1,0.5)*box_w #1/10---1/2
			adv_size_h=random.uniform(0.1,0.5)*box_h
			
			box_s_x=float(box[0])-box_w/2
			box_s_y=float(box[1])-box_h/2
			box_e_x=float(box[0])+box_w/2
			box_e_y=float(box[1])+box_h/2
			#random location
			adv_s_x=int(math.floor(random.uniform(box_s_x,box_e_x-adv_size_w)*img_w))
			adv_e_x=int(math.floor(adv_s_x+adv_size_w*img_w))
			adv_s_y=int(math.floor(random.uniform(box_s_y,box_e_y-adv_size_h)*img_h))
			adv_e_y=int(math.floor(adv_s_y+adv_size_h*img_h))
			#gray replace
			img2[adv_s_y:adv_e_y,adv_s_x:adv_e_x]=128
    
    return img2

#直接对抗块替换，尽可能放于目标中心位置
class listDataset3(Dataset):  

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if not self.train:
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
        else:
            #print('===============imgpath==================',imgpath)  #连续执行一个batch的img！！！！！！！！！！！1
            img = Image.open(imgpath).convert('RGB')
            
            #if self.shape:
            #    img = img.resize(self.shape)
            img_name=imgpath.split('/')[-1].strip()
            #labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            labpath = imgpath.replace('/img4/', '/img4_pred_label/').replace('.jpg', '.txt')
            #print('===============labpath==================',labpath)
            #1、随机块替换,输入：原图像、及label路径；输出替换后图像及对应mask
            #print('====================================================',img.size[0],img.size[1])
            
            img_rep=rand_repl2(img,labpath)
            img=Image.fromarray(img_rep.astype('uint8')).convert('RGB')
            #这里需要注意的是上面是Image操作的，而修复需要转为opencv的
        
            #2、块修复,输入如上输出，返回修复后图像
            #img2 = cv2.imread(imgpath)
            #img=patch_repa(img_rep,img_mask)
            
            #原label不变
            
            label = torch.zeros(50*5)
            #if os.path.getsize(labpath):
            #tmp = torch.from_numpy(np.loadtxt(labpath))
            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
            except Exception:
                tmp = torch.zeros(1,5)
            #tmp = torch.from_numpy(read_truths(labpath))
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            #print('labpath = %s , tsz = %d' % (labpath, tsz))
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return (img_name,img, label)

#随机块替换,随机块大小指定为占目标大小的的1/6-1/2，在范围内随机取位置/大小
#对抗块替换：/home/zgz/adversarial_samples/adversarial-yolo-master/gen_adv_voc3/100_3.png
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])   分别*255获取三通道各像素均值
def rand_repl2(img,labpath):
    
    img2=np.array(img)
    img_w=img.size[0]
    img_h=img.size[1]
    
    lines=codecs.open(labpath,'r').readlines()
    for line in lines:
		
        box=line.split(' ')[1:] #取出相对的x\y\w\h，注意这里是字符类型
        box_w=float(box[2])
        box_h=float(box[3])

        cls_id=line.split(' ')[0]
        adv_size_w=random.uniform(0.1,0.3)*box_w #1/10---1/3
        adv_size_h=random.uniform(0.1,0.3)*box_h
        
        box_s_x=float(box[0])-adv_size_w/2
        box_s_y=float(box[1])-adv_size_h/2
        box_e_x=float(box[0])+adv_size_w/2
        box_e_y=float(box[1])+adv_size_h/2
        
        adv_s_x=int(math.floor(box_s_x*img_w))
        adv_e_x=int(math.floor(adv_s_x+adv_size_w*img_w))
        adv_s_y=int(math.floor(box_s_y*img_h))
        adv_e_y=int(math.floor(adv_s_y+adv_size_h*img_h))
        
        img_adv_sw=int(adv_e_x-adv_s_x)
        img_adv_sh=int(adv_e_y-adv_s_y)
        
        img2[adv_s_y:adv_e_y,adv_s_x:adv_e_x]=128
    
    return img2

class MyDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train and index % 64== 0:
            if self.seen < 4000*64:
               width = 13*32
               self.shape = (width, width)
            elif self.seen < 8000*64:
               width = (random.randint(0,3) + 13)*32
               self.shape = (width, width)
            elif self.seen < 12000*64:
               width = (random.randint(0,5) + 12)*32
               self.shape = (width, width)
            elif self.seen < 16000*64:
               width = (random.randint(0,7) + 11)*32
               self.shape = (width, width)
            else: # self.seen < 20000*64:
               width = (random.randint(0,9) + 10)*32
               self.shape = (width, width)

        if self.train:
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
        else:
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
    
            #labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            file=imgpath.split('/')[-1]
            labpath = imgpath.replace(file,'yolo-labels/')+file.replace('.jpg', '.txt').replace('.png','.txt')
            label = torch.zeros(50*5)
            #if os.path.getsize(labpath):
            #tmp = torch.from_numpy(np.loadtxt(labpath))
            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
            except Exception:
                tmp = torch.zeros(1,5)
            #tmp = torch.from_numpy(read_truths(labpath))
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            #print('labpath = %s , tsz = %d' % (labpath, tsz))
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return (img, label)

class MyDataset2(Dataset):  #更改label来自图片名字

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train and index % 64== 0:
            if self.seen < 4000*64:
               width = 13*32
               self.shape = (width, width)
            elif self.seen < 8000*64:
               width = (random.randint(0,3) + 13)*32
               self.shape = (width, width)
            elif self.seen < 12000*64:
               width = (random.randint(0,5) + 12)*32
               self.shape = (width, width)
            elif self.seen < 16000*64:
               width = (random.randint(0,7) + 11)*32
               self.shape = (width, width)
            else: # self.seen < 20000*64:
               width = (random.randint(0,9) + 10)*32
               self.shape = (width, width)

        if not self.train:
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
        else:
            img = Image.open(imgpath).convert('RGB')
            ori_W,ori_H=img.size
            var_size_W,var_size_H=ori_W/416.0,ori_H/416.0
            if self.shape:
                #print('img   size  --------------------',img.size[0],img.size[1],self.shape)
                img = img.resize((416,416))
                #print('img   size  ------------------========--',img.size[0],img.size[1])
    
            #labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            file=imgpath.split('/')[-1]
            #labpath = imgpath.replace(file,'yolo-labels/')+file.replace('.jpg', '.txt').replace('.png','.txt')
            #print('==========================fiel===============',file)
            
            truths=file.split('__')[1:-1]
            clos=len(truths)/4
            truths=np.array(truths).reshape(clos, 4)
            
            W,H=img.size
            new_truths=[]
            for k in range(clos):
                box=(float(truths[k][0])/var_size_W, float(truths[k][1])/var_size_H, float(truths[k][2])/var_size_W, float(truths[k][3])/var_size_H)
                new_truths.append(convert(W, H, box))
            
            #print('=============W,H=============new_truths===============',W,H,new_truths)
            
            label = torch.zeros(50*5)
            tmp = torch.from_numpy(np.array(new_truths).astype('float32'))
            #tmp = torch.from_numpy(read_truths(labpath))
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            #print('labpath = %s , tsz = %d' % (labpath, tsz))
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        #print('==========================label size===============',label.shape)
        return (img, label)

def convert(W, H, box):  #size表示整体图像的w、h，内框box传进来的是(x,y,w,h)
    dw = 1./W
    dh = 1./H
    x = (box[0] + box[2]+box[0])/2.0
    y = (box[1] + box[3]+box[1])/2.0
    w = box[2]
    h = box[3]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [0,x,y,w,h]

'''
#坐标格式变换，box(xmin,xmax,ymin,ymax)
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
'''
