import os
import codecs
from PIL import Image

rd1='/home/zgz/pytorch-yolo2-master/9.29_LaVAN_test1'
rd2='/home/zgz/pytorch-yolo2-master/9.16_LaVAN_test2'
rd3='/home/zgz/pytorch-yolo2-master/9.16_NDSS_TrojanNN_test'

wd1='/home/zgz/pytorch-yolo2-master/9.29_LaVAN_test_label1/'
wd2='/home/zgz/pytorch-yolo2-master/9.16_LaVAN_test_label2/'
wd3='/home/zgz/pytorch-yolo2-master/9.16_NDSS_TrojanNN_test_label/'

for root,dirs,files in os.walk(rd1):
    for f in files:
        path=os.path.join(root,f)
        img=Image.open(path).convert('RGB')
        #print(img.size)
        wri_path=wd1+f.replace('.JPEG','.txt').replace('.png','.txt').replace('.jpg','.txt')
        #x_s=210/299
        x_s=180/256
        y_s=x_s
        #w=50/299
        w=50/256
        h=w

        wf=codecs.open(wri_path,'w')
        wf.write('0 '+str(x_s+w/2)+' '+str(y_s+h/2)+' '+str(w)+' '+str(h))

