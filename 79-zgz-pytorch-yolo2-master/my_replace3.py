#encoding=utf-8
#生成含有随机大小、随机数量对抗块的随即大小对抗图像

#这一个文件只针对于测试集的生成，选出了1000张测试块，为了将所有的块都用上，所以将块放在外循环，干净图像放在内循环
#每次遍历一个对抗块，随机选取10张干净图像进行贴附，每次选取干净图像时进行shuffle

#注意这里生成时的流程是：先复制干净图像到9.18_adv_img_test，然后读/操作/保存同在这一个文件夹

import os
from PIL import Image
import numpy as np
from scipy import misc
import random

patch_dir='/home/zgz/pytorch-yolo2-master/9.18_adv_patch_test'
#img_dir='/home/zgz/pytorch-yolo2-master/voc_test'
img_dir='/home/zgz/pytorch-yolo2-master/9.18_adv_img_test'
#img_dir='/home/zgz/pytorch-yolo2-master/9.18_adv_img_test'
adv_save='/home/zgz/pytorch-yolo2-master/9.18_adv_img_test/'


classes = ['aeroplane','bicycle','bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow','diningtable',
        'dog', 'horse', 'motorbike', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor', 'person']

fadv_name=''
fadv_cnt=0
cnt=0
for root,dirs,files in os.walk(patch_dir):   
    
    for f in files:
        
        img_adv=Image.open(os.path.join(root,f))
        
        num_cnt=0
        num=10
        for i in range(0,10):   #每个块选择10张干净图像
            
            for root2,dirs2,files2 in os.walk(img_dir):

                random.shuffle(files2)
                
                if num_cnt>=num:#or cnt>4999:
                    break

                for f2 in files2:
                    if num_cnt>=num:# or cnt>4999:
                        break
                    
                    img=Image.open(os.path.join(root2,f2))
                    img_size=img.size
                    
                    img=np.array(img)
                    
                    img__w=random.randint(10,90)
                    if random.randint(0,10)>5:
                        img__h=random.randint(10,90)
                    else:
                        img__h=img__w
                        
                    img_adv=img_adv.resize((img__w,img__h))
                    patch_w=img_adv.size[0]
                    patch_h=img_adv.size[1]
                    
                    x=np.random.randint(0,img_size[0]-patch_w)
                    y=np.random.randint(0,img_size[1]-patch_h)
                    
                    img_name=f2
                    #if random.randint(0,10)>3:

                    img[y:y+patch_h,x:x+patch_w]=np.array(img_adv)

                    img_name=str(random.randint(0,10000))+'_'+f2.split('/')[-1].split('.')[0]+'__'+str(x)+'__'+str(y)+'__'+str(patch_w)+'__'+str(patch_h)+'.jpg'
                    
                    misc.imsave(adv_save+img_name,img)
                    
                    num_cnt=num_cnt+1
                    cnt=cnt+1

                
                

