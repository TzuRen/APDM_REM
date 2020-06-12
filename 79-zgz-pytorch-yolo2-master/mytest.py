import os
import codecs
from PIL import Image

rd1='/home/zgz/pytorch-yolo2-master/9.29_LaVAN_test1'
rd2='/home/zgz/pytorch-yolo2-master/mytest'
wd='/home/zgz/pytorch-yolo2-master/9.29_LaVAN_test3/'

for r,ds,fs in os.walk(rd1):
    for f in fs:
        s=0
        for r2,ds2,fs2 in os.walk(rd2):
            for f2 in fs2:
                if f.split('_')[0:5]==f2.split('_')[0:5]:
                    s=1
                    break
        if s==0:
            img_path=os.path.join(r,f)
            img=Image.open(img_path)
            img.save(wd+f)


