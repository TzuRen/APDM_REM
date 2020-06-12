import codecs
from PIL import Image
import random

wf=codecs.open('./2007_test.txt','r')
lines=wf.readlines()

ims='/home/zgz/pytorch-yolo2-master/voc_test/'

for line in lines:
    line=line.strip()
    im=Image.open(line)
    la=line.replace('JPEGImages','labels').replace('jpg','txt').replace('png','txt')
    f=codecs.open(la,'r').readlines()
    s=''
    for li in f:
        s=s+li.split(' ')[0]+'_'
#    im=Image.open(line)
    #print(line)
    im.save(ims+s+line.split('/')[-1])
