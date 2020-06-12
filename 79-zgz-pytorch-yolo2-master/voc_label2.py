import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import codecs
import shutil
from PIL import Image

'''
rf=codecs.open('./voc_train.txt','r').readlines()
dst='/home/zgz/pytorch-yolo2-master/voc_train_lable/'
wd='/home/zgz/pytorch-yolo2-master/voc_train_img/'

for line in rf:
    line=line.strip()
    #pa=line.replace('JPEGImages','labels').replace('jpg','txt').replace('png','txt')
    #shutil.copyfile(pa,dst+pa.split('/')[-1])
    im=Image.open(line)
    im.save(wd+line.split('/')[-1])


'''
sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ['helmet']

def convert(size, box):   #具体原理？？？
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

def convert_annotation(xml_path, txt_path):   #1.解析xml到labels下的txt：类别+坐标
    in_file = open(xml_path,'r')
    out_file = open(txt_path, 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    print(xml_path,size[0])
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

xml_dir='./pic1_gai/xml'

for root,dirs,files in os.walk(xml_dir):
    for f in files:
        xml_path=os.path.join(root,f)
        txt_path='./pic1_gai/txt/'+f.replace('xml','txt')
        convert_annotation(xml_path, txt_path)

'''
wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:           #2.于根目录下copy相应的train/val完整路径数据集
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        convert_annotation(year, image_id)
    list_file.close()
'''
