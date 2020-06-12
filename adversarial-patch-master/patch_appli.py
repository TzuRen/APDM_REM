import os
from PIL import Image
import numpy as np

img_from='/home/zgz/adversarial_samples/adversarial-patch-master/ori'
img_to='/home/zgz/adversarial_samples/adversarial-patch-master/ori_patch/'

patch=np.array(Image.open('patch.jpg').convert('RGB').resize((60,60)))

for ro,ds,fs in os.walk(img_from):
    for f in fs:
        img1=np.array(Image.open(os.path.join(ro,f)))
        img1[230:290,230:290]=patch
        Image.fromarray(img1).save(img_to+f)
