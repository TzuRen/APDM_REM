import os
from PIL import Image
import random

rd='/home/zgz/pytorch-yolo2-master/pic_patch2'
wd='/home/zgz/pytorch-yolo2-master/pic_patch3'

rd='/home/zgz/adversarial_samples/adversarial-yolo-master2/gen_adv_darkvoc_0'
wd='/home/zgz/pytorch-yolo2-master/9.18_adv_patch_train/'
wd2='/home/zgz/pytorch-yolo2-master/9.18_adv_patch_test/'
i=0
for root ,dirs,files in os.walk(rd):
    for f in files:
        a=random.randint(1,10)
        if a<5:
            continue
        if i>299:
            break

        
        path=os.path.join(root,f)
        im=Image.open(path)

        
        if i>150 and i<201:
            path=wd2+f
            im.save(path)
        else:
            path=wd+f#path.replace('pic_patch2','pic_patch3')
            im.save(path)
        i=i+1

