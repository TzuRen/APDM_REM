#  -*- coding: utf-8 -*-

import os
import cv2 as cv
import numpy as np
import sys
import codecs
import math
from math import *
import random
from PIL import Image

def dumpRotateImage(img, degree, pt1, pt3):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut = imgRotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])]
    height, width = imgOut.shape[:2]
    return imgOut

#计算图像三通道熵及均值
def entro(img):
    
    #img=Image.open(imgpath)
    img=img.convert('RGB')
    r,g,b=img.split()
    
    #r,g,b=imgpath.split()
    #r=imgpath.split()
    #g=r
    #b=r
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


'''235 data目录下处理文件'''
#root_dir = '/home/zgz/picture_txt'
#root_dir=r'./inter_pic/t2'
#root_dir=r'./inter_pic/t2'
root_dir=r'./inter_pic/3999_60'
root_dir='/home/zgz/adversarial_samples/adversarial-yolo-master/ad_p_1'
#保存截取的图片路径
#cut_img_output_dir = '/home/zgz/intercept_picture'
cut_img_output_dir=r'./inter_pic/3999_60_inter'
cut_img_output_dir=r'/home/zgz/adversarial_samples/adversarial-yolo-master/ad_p_2_inter1'
#保存输出的文件名称
#path_text = r'C:\Users\Enjoy\Desktop\picture\58_inter_txt3.txt'
#total_cut_img_num = 0

#file_object = codecs.open(path_text, 'w', 'utf-8')

dev_entor=np.zeros((300,300))

for path,d,filelist in os.walk(root_dir):

    for filename in filelist:

        full_path = os.path.join(path, filename)
        #有隐藏文件，需要过滤
        if os.path.splitext(full_path)[1] == '.png' or os.path.splitext(full_path)[1] == '.jpg':
            prefix_filename = filename.split('.')[0]
            #pre_image = cv.imread(os.path.join(path,filename))
            pre_image = Image.open(os.path.join(path,filename))
            size=np.array(pre_image)
            per=0
        
            step=5
            dev=45
            num=((len(size[0])-dev)/dev+1)*dev
            
            #二维矩阵
            w,h=0,0
            
            for i in range(0,len(size[0])-dev,step):
                w=0
                for j in range(0,len(size[1])-dev,step):
                    box=(i,j,i+dev,j+dev)
                    cut_img = pre_image.crop(box)
                    #分割好了直接计算其对应熵值，然后存储到二维矩阵，以黑白绘画出来观察
                    ent=entro(cut_img)
                    output_filename = prefix_filename + '_' +str(ent)+'_'+str(i)+'_'+str(j)+ ".png"

                    
#                     if ent<7.0:
#                         dev_entor[w][h]=255
#                     else:
#                         dev_entor[w][h]=0
#                     #print(dev_entor[w][h])
#                     w+=1
                    
#                 h+=1
             
#             plt.imshow(Image.fromarray(dev_entor))
#             plt.show()

                    #保存截取图片
                    img_save_path = os.path.join(cut_img_output_dir, output_filename)
                    cut_img.save(img_save_path)
                  
                
                    
#             while per<50:

#                 a = random.randint(1,350)
#                 b = random.randint(1,350)
#                 c = a+40
#                 d = b+40

#                 point1 = (a, b)

#                 point3 = (c, d)

#                 #cut_img = dumpRotateImage(pre_image, degrees(atan2(0, point3[0] - point1[0])), point1,point3)
#                 box=(a,b,c,d)
#                 cut_img = pre_image.crop(box)
#                 output_filename = prefix_filename + '_' + str(per) + ".jpg"

#                 # 保存截取图片
#                 img_save_path = os.path.join(cut_img_output_dir, output_filename)
                
#                 per+=1
#                 cut_img.save(img_save_path)
                #if cut_img.shape[0]>0:
                #cv.imwrite(img_save_path, cut_img)

                # 保存截取图片对应的text
#                 #print (text_content)
#                 write_text = str(img_save_path) + ' ' + str(text_content)
#                 file_object.write(write_text)
#                 #file_object.write('\n')
#                 line = text_file.readline()
