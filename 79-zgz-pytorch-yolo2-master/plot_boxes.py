#encoding=utf-8
#根据传入的图像及boxes信息，进行画框，这里的boxes为相对起始x/y及w/h
import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import codecs
import struct # get_image_size
import imghdr # get_image_size

def plot_boxes(img, lines, savename=None, class_names=None):
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for line in lines:
        box = line.split(' ')[1:]
        x1 = (float(box[0])-float(box[2])/2.0) * width
        y1 = (float(box[1])-float(box[3])/2.0) * height
        x2 = (float(box[0]) + float(box[2])/2.0) * width
        y2 = (float(box[1]) + float(box[3])/2.0) * height

        rgb = (255, 0, 0)
        if class_names:
            
            cls_id = int(line.split(' ')[0])
            print('%s' % (class_names[cls_id]))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            rgb = (red, green, blue)
            
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        
        draw.rectangle([x1, y1, x2, y2], outline = rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    #return img

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

imgd='video_dispaly/img4_adv' #原始图像
labd='video_dispaly/img4_adv_repl_gray_pred_label/'  #需要画框的label
imgsave='video_dispaly/img4_adv_dealed_pred/' #画完之后的保存

namesfile='data/voc.names'
class_names = load_class_names(namesfile)

for root,ds,fs in os.walk(imgd):
    for f in fs:
        img=Image.open(os.path.join(root,f)).convert('RGB')
        
        lines=codecs.open(labd+f.replace('.jpg','.txt')).readlines()
        plot_boxes(img, lines,imgsave+f,class_names)
        
            
