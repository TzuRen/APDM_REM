#encoding=utf-8
#生成含有固定大小60*60、固定数量1的对抗块的对抗图像
import os
from PIL import Image
import numpy as np
from scipy import misc

patch_dir='/home/zgz/pytorch-yolo2-master/pic_patch'
img_dir='/home/zgz/pytorch-yolo2-master/my_VOC07'

adv_save='/home/zgz/pytorch-yolo2-master/my_adv_rep/'

classes = ['aeroplane','bicycle','bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow','diningtable',
        'dog', 'horse', 'motorbike', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor', 'person']

img_size=300
adv_size=60

for root,dirs,files in os.walk(img_dir):
    for f in files:
#        print(f)
        idx=f.split('_')[:-1]
        s=[classes[int(i)] for i in idx]
 #       print(s)
        img=Image.open(os.path.join(root,f))
#        print(img.size)
        img=img.resize((img_size,img_size))
 #       print('img:',img.size[1],img.size)
        img=np.array(img)
        #print(img.shape)

        for root2,dirs2,files2 in os.walk(patch_dir):
            i=0
            if i<11:
                #img2=img
                for f2 in files2:
                    i=np.random.randint(0,60000)
                    j=0
                    if j!=i:
                        continue
                #if i<11:
                    if f2.split('_')[0] not in s:
                        img_adv=Image.open(os.path.join(root2,f2))
                    #print('adv:',img_adv.size[1])
                        x=np.random.randint(0,img_size-adv_size)
                        y=np.random.randint(0,img_size-adv_size)
                        img2=img
                        img2[x:x+adv_size,y:y+adv_size]=np.array(img_adv)
                    #img=Image.fromarray(np.uint8(img))
                        misc.imsave(adv_save+f.split('.')[0]+'__'+str(x)+'__'+str(y)+'__.jpg',img2)
                    
                    i=i+1
                    break
                break
