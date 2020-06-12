import os
import codecs

rd='/home/zgz/pytorch-yolo2-master/9.18_19_1500_adv_test_rep'
rd='/home/zgz/pytorch-yolo2-master/9.16_LaVAN_test2'
rd='/home/zgz/pytorch-yolo2-master/9.16_NDSS_TrojanNN_test'
rd='/home/zgz/pytorch-yolo2-master/9.29_LaVAN_test1'
rd='/home/zgz/pytorch-yolo2-master/9.18_14_1500_adv_test_gray_repla1'
rd='/home/zgz/adversarial_samples/adversarial-yolo-master2/10.14_oriimg_detec'
#rd='/home/zgz/pytorch-yolo2-master/9.18_14_1500_adv_test_gray_repla2'
#rd='/home/zgz/pytorch-yolo2-master/9.29_LaVAN_test_label1.txt'
#wf=codecs.open('/home/zgz/pytorch-yolo2-master/9.16_NDSS_TrojanNN_test.txt','w')
#wf=codecs.open('/home/zgz/pytorch-yolo2-master/9.29_LaVAN_test1.txt','w')
#wf=codecs.open('/home/zgz/pytorch-yolo2-master/9.29_LaVAN_test_label1.txt','w')
#wf=codecs.open('/home/zgz/pytorch-yolo2-master/9.18_14_1500_adv_test_gray_repla1.txt','w') 
#wf=codecs.open('/home/zgz/pytorch-yolo2-master/9.18_14_1500_adv_test_gray_repla2.txt','w') 
wf=codecs.open('/home/zgz/adversarial_samples/adversarial-yolo-master2/10.14_oriimg_detec.txt','w')

for root,dirs,fs in os.walk(rd):
    for f in fs:
        path=os.path.join(root,f)
        wf.write(path+'\n')
