#  -*- coding: utf-8 -*-
#2019.10.9
#只利用熵值阈值，进行灰度块替换对抗图像
import os
import cv2 as cv
import numpy as np
import sys
import codecs
import math
from math import *
import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn as nn

#计算图像三通道熵及均值
def entro(imgpath):
    
    #img=Image.open(imgpath)
    #r,g,b=img.split()
    
    r,g,b=imgpath.split()
    rn=np.array(r)
    gn=np.array(g)
    bn=np.array(b)

    tmp = []
    tmp2 = []
    tmp3 = []
    for i in range(256):
        tmp.append(0)
        tmp2.append(0)
        tmp3.append(0)

    val = 0
    k = 0
    res = 0
    val2 = 0
    k2 = 0
    res2 = 0
    val3 = 0
    k3 = 0
    res3 = 0

    img =rn
    img2 =gn
    img3 =bn

    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)
            val2 = img2[i][j]
            tmp2[val2] = float(tmp2[val2] + 1)
            k2 =  float(k2 + 1)
            val3 = img3[i][j]
            tmp3[val3] = float(tmp3[val3] + 1)
            k3 =  float(k3 + 1)

    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
        tmp2[i] = float(tmp2[i] / k2)
        tmp3[i] = float(tmp3[i] / k3)

    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        if(tmp[i]!= 0):
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
        if(tmp2[i] == 0):
            res2 = res2
        if(tmp2[i]!= 0):
            res2 = float(res2 - tmp2[i] * (math.log(tmp2[i]) / math.log(2.0)))  
        if(tmp3[i] == 0):
            res3 = res3
        if(tmp3[i]!= 0):
            res3 = float(res3 - tmp3[i] * (math.log(tmp3[i]) / math.log(2.0)))

    entro=round((res+res2+res3)/3,2)  #保留2位小数

    return entro

#计算图像三通道熵及均值
def entro_gray(imgpath):
    
    r=imgpath
    rn=np.array(r)
    
    tmp = []
    for i in range(256):
        tmp.append(0)
        
    val = 0
    k = 0
    res = 0

    img =rn
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)
            
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
        
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        if(tmp[i]!= 0):
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
        
    entro=round(res,2)  #保留2位小数

    return entro
    
#高亮图像块内值赋值0.5变为灰度，周围按整体图像大小填充0
#图像块像素值填充
def generate_patch(type,w,h):
    
    if type == 'gray':
        #生成一个前面维度的全部值为0.5的tensor
        adv_patch_cpu = torch.full((3, w,h), 0.5)
    elif type == 'random':
        adv_patch_cpu = torch.rand((3, w,h))
    elif type == 'black':
        adv_patch_cpu = torch.full((3, w,h), 1)

    return adv_patch_cpu

def gen_img(img,color,box,size0,size1):
    
    i,j,w,h=box
    adv=generate_patch(color,w-i,h-j)
    
    #上下左右分别pad的大小
    pads,padx,padz,pady=j,size1-h,i,size0-w  #原i,size0-w,j,size1-h ，进行顺时针旋转90度   
    #print(pads,padx,padz,pady)

    if color=='gray':
        mypad = nn.ConstantPad2d((int(padz), int(pady), int(pads), int(padx)), 0)
    if color=='black':
        mypad = nn.ConstantPad2d((int(padz), int(pady), int(pads), int(padx)), 0)
    #左右上下四个维度分别按指定int大小填充相应个数0，使图像块填充后大小和原图像大小相同
    #填充的值为0，在两者融合时过滤掉0值
    #分别对应同等大小的图像块和label_id(扩维后得)，将两者相乘表示以id过滤图像块
    adv = mypad(adv)

    loader = transforms.Compose([
    transforms.ToTensor()]) 
    
    image = loader(img)
    #print(image.shape,adv.shape)
    if color=='gray':
        img= torch.where((adv == 0), image, adv)
    if color=='black':
        img= torch.where((adv == 0), image, adv)

    return img
    
'''235 data目录下处理文件'''
#读取图片
root_dir='/home/zgz/pytorch-yolo2-master/9.18_14_1500_adv_test'
root_dir=r'./crop001251'
#保存替换后的图片路径
replace_dir='/home/zgz/pytorch-yolo2-master/9.18_14_1500_adv_test_gray_repla1/'
replace_dir='./crop001251/'
step=3
dev=30
#size[0]=416
num=((416-dev)//step+1)*dev
num2=(416-dev)//step+1
dev_entor=np.zeros((num2,num2))

unloader = transforms.ToPILImage()

for path,d,filelist in os.walk(root_dir):

    for filename in filelist:

        full_path = os.path.join(path, filename)
        print(full_path)
        #有隐藏文件，需要过滤
        if os.path.splitext(full_path)[1] == '.png' or os.path.splitext(full_path)[1] == '.jpg':
            
            dev_entor=np.zeros((num2,num2))
            
            prefix_filename = filename.split('.')[0]
            #pre_image = cv.imread(os.path.join(path,filename))
            image = Image.open(os.path.join(path,filename)).convert('RGB')
            pre_image = Image.open(os.path.join(path,filename))#.convert('L')
#             plt.imshow(pre_image)
#             plt.show()
            size=np.array(pre_image)
            per=0
        
            #二维矩阵
            w,h=0,0
            
            for i in range(0,len(size[0])-dev,step):
                w=0
                for j in range(0,len(size[1])-dev,step):
                    box=(i,j,i+dev,j+dev)
                    cut_img = pre_image.crop(box)
                    output_filename = prefix_filename + '_' + str(i)+'_'+str(j) +'_'+str(i*num+j)+ ".png"

                    #分割好了直接计算其对应熵值，然后存储到二维矩阵，以黑白绘画出来观察
                    ent=entro(cut_img)
                    
                    #只利用熵值，灰度块替换，查看检测效果
                    if ent>7.289 and ent<7.301:
                        
                        image=gen_img(image,'gray',box,len(size[0]),len(size[1]))
                        image=unloader(image)
                            
        image.save(replace_dir+'gray_'+filename)
