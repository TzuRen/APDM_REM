#encoding=utf-8

import os
from PIL import Image
import codecs

img_dir='/home/zgz/pytorch-yolo2-master/VOCdevkit/VOC2007/JPEGImages'
lables_dir='/home/zgz/pytorch-yolo2-master/VOCdevkit/VOC2007/labels/'

save_img='/home/zgz/pytorch-yolo2-master/my_VOC/19/'
save_lab='/home/zgz/pytorch-yolo2-master/my_VOC/19_lables/'
#fw=codecs.open(save_lab,'w')

if not os.path.exists(save_img):
    os.mkdir(save_img)

if not os.path.exists(save_lab):
    os.mkdir(save_lab)

for root,dirs,files in os.walk(img_dir):
    for f in files:
        
        img=Image.open(os.path.join(root,f))
        lables=lables_dir+f.replace('.jpg','.txt')
        fl=codecs.open(lables,'r')
        lines=fl.readlines()
        s=''
        l=''
        for line in lines:
            s=s+line.split(' ')[0]+'_'
            l=l+line
        if '19' in s.split('_'):
            img.save(save_img+f)
            fw=codecs.open(save_lab+f.replace('.jpg','.txt'),'w')
            fw.write(l)

