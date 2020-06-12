import os
import codecs

rd='/home/zgz/pytorch-yolo2-master/9.18_14_1500_adv_test_gray_repla2'

wf=codecs.open('/home/zgz/pytorch-yolo2-master/9.18_14_1500_adv_test_gray_repla2.txt','w')

for root,dirs,fs in os.walk(rd):
    for f in fs:
        path=os.path.join(root,f)
        wf.write(path+'\n')
