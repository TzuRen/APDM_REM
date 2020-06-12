#encoding=utf-8
#随机筛选对抗块到pic_patch3,来源于两部分
#1.原pic_patch2选出2000张
#2.生成的对抗块共20类，每类从200-500轮选择100张
#注意第二部分选择的对抗块需要更改名称格式，包括类别名加对抗块大小w\h
#且原size=(300,300)，需要随机resize

#保存到9.3_adv_patch_all，来源于两部分
#少量500张于pic_patch2
#生成的对抗块选择2000张，来源gen_adv_all6_*

import os
from PIL import Image
import random

rd1='/home/zgz/pytorch-yolo2-master/pic_patch2'
#rd1='/home/zgz/adversarial_samples/adversarial-yolo-master/saved_patches'
#rd1='/home/zgz/adversarial_samples/adversarial-yolo-master/gen_adv_voc19'
#已有0、2、3、4、5、7、14

rd1='/home/zgz/adversarial_samples/adversarial-yolo-master2/gen_adv_all6_'

wd1='/home/zgz/pytorch-yolo2-master/random_adv_2000/'
wd='/home/zgz/pytorch-yolo2-master/pic_patch4/'
wd='/home/zgz/pytorch-yolo2-master/9.3_adv_patch_all/'

class_=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus','car','cat','chair', 'cow','diningtable', 'dog', 'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
i=0
for k in range(2,3):
    print('-------------',k,i) 
    i=0
    rd1='/home/zgz/adversarial_samples/adversarial-yolo-master2/gen_adv_all6_'
    rd1=rd1+str(k)
    print(rd1)
    for root,dirs,files in os.walk(rd1):
         
        for f in files:
            if i>100:
                break
        #i=i+1
            j=random.randint(0,1000)
            if j<200:
                continue
            img_p=os.path.join(root,f)
        #epoch=int(img_p.split('/')[-1].split('_')[0])
        #epoch=int(img_p.split('/')[-1].split('_')[-1].split('.')[0])
      #  classid=int(img_p.split('_')[1].split('.')[0])
        #if epoch>500 or epoch<100:
        #    continue
            im=Image.open(img_p)

            w=random.randint(20,70)
            h=random.randint(20,70)
            im=im.resize((w,h))
            im_p=wd+class_[k]+'_'+str(w)+'_'+str(h)+'_'+str(random.randint(0,10000))+'.png'
            #im_p1=wd+f
            im.save(im_p)
        #im.save(im_p)
            i=i+1
