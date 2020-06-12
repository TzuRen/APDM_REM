import os
import codecs

img_dir='/home/zgz/pytorch-yolo2-master/9.18_adv_img_train'
to_file=codecs.open('./9.18_adv_img_train.txt','w')

for root,dirs,files in os.walk(img_dir):
    for file in files:
        img_path=os.path.join(root,file)
        to_file.write(img_path+'\n')
