#  -*- coding: utf-8 -*-
#2019.10.9
#在熵值阈值检测基础上，加上差值阈值过滤，然后利用灰度块替换对抗块
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
  
#图像块内像素差值计算，返回差值在一定范围内的个数，即这里的返回差值小于20的像素个数
#传入为图像块
def pixel_cut_gray(cut_img):
            
    rn=np.array(cut_img)

    #方法1：有效果，且较明显
    m,n=cut_img.size[0],cut_img.size[1]

    entor1=np.zeros((m,n))
    cnt=0

    for i in range(m-1):
        for j in range(n-1):
            entor1[i][j]=np.sqrt(pow(rn[i,j]-rn[i+1,j],2)+pow(rn[i,j]-rn[i,j+1],2))

    #print(type(a),a,b,c,a-b)   在这里浪费了我大量时间！！这里的数据是uint8类型的，其范围为0-255，
    #超出范围之后再次循环取值，所以不能直接运算，需要转换处理

    #print(88-97,c,rn[0,1],rn[0,2],a,b,a-b,rn[0,1]-rn[0,2],pow(a,2),entor1[0][1])
    mean_entor=entor1
    count=mean_entor[mean_entor<20]
    return count.size
    
'''235 data目录下处理文件'''
#读取图片
root_dir='/home/zgz/pytorch-yolo2-master/9.18_14_1500_adv_test'

#保存替换后的图片路径
replace_dir='/home/zgz/pytorch-yolo2-master/9.18_14_1500_adv_test_gray_repla2/'

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
            pre_image = Image.open(os.path.join(path,filename)).convert('L')
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
                    ent=entro_gray(cut_img)
                    
                    #只利用熵值，灰度块替换，查看检测效果
                    if not ent<7.35:
                         
                        ct=pixel_cut_gray(cut_img)
                        if ct>215:
                            
                            image=gen_img(image,'gray',box,len(size[0]),len(size[1]))
                            image=unloader(image)
        image.save(replace_dir+filename)
