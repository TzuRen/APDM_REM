#encoding=utf-8
#生成含有随机大小、随机数量对抗块的随即大小对抗图像
import os
from PIL import Image
import numpy as np
from scipy import misc
import random

#patch_dir='/home/zgz/pytorch-yolo2-master/pic_patch2'

#patch_dir='/home/zgz/pytorch-yolo2-master/pic_patch4' #加入了真实对抗块
#patch_dir='/home/zgz/pytorch-yolo2-master/9.3_adv_patch_all'
patch_dir='/home/zgz/pytorch-yolo2-master/9.18_adv_patch_train'
patch_dir='/home/zgz/pytorch-yolo2-master/9.29_LaVAN_adv_patch/net_dom'
#img_dir='/home/zgz/pytorch-yolo2-master/my_VOC07'   #横向图 500*375 或（纵向图） 375*500
img_dir='/home/zgz/pytorch-yolo2-master/voc_train'
img_dir='/home/zgz/pytorch-yolo2-master/imagenet_randsele_250'
#img_dir='/home/zgz/pytorch-yolo2-master/voc_test'
#adv_save='/home/zgz/pytorch-yolo2-master/my_adv_rep6/'
#adv_save='/home/zgz/pytorch-yolo2-master/my_adv_rep6_test/'
adv_save='/home/zgz/pytorch-yolo2-master/9.29_LaVAN_test1/'

classes = ['aeroplane','bicycle','bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow','diningtable',
        'dog', 'horse', 'motorbike', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor', 'person']

#img_w=np.random.randint(350,500)
#img_h=np.random.randint(350,500)
#img_size=(img_w,img_h)

#adv_size=60  这里的图像块大小是读取时根据文件名解析出来的
fadv_name=''
fadv_cnt=0
for root,dirs,files in os.walk(img_dir):   
    
    for f in files:
        
        idx=f.split('_')[:-3]  # 类别+w+h+randint
        s=[classes[int(i)] for i in idx]
 #       print(s)
        img=Image.open(os.path.join(root,f))
        img_size=img.size
        if img_size[0]<230 or img_size[1]<230:
            continue
        img=np.array(img)
        adv_name1=adv_save+f.split('.')[0] #最终对抗图像的名字
                
        #for i in range(11): #一张图像生成11张对抗图像 这一步直接这样做是有问题的，img2=img，更改img2其img也会变
        
        img2=img
        num=1#np.random.randint(0,4)
        adv_name=adv_name1+'__'
        ss=''

        num_cnt=0
            
        for root2,dirs2,files2 in os.walk(patch_dir):
            
            if num_cnt>=num:
                break

            for f2 in files2:
                if num_cnt>=num:
                    break
                if random.randint(0,100)>20:
                    continue

                '''
                if f2 in fadv_name:
                    fadv_cnt=fadv_cnt+1
                    if fadv_cnt>4000000:
                        fadv_cnt=0
                        print('--------------------------------------------fadv---claer---')
                        fadv_name=''
                    continue
                '''
                img_adv=''
                if f2.split('_')[0] not in s:  #确保块的类别与图像内类别不同
                    img_adv=Image.open(os.path.join(root2,f2))
                    #print('adv:',img_adv.size[1])
                    #img_adv=img_adv.resize((random.randint(20,90),random.randint(20,90)))
                    '''
                    img__w=random.randint(20,60)
                    if random.randint(0,10)>7:
                        img__h=random.randint(20,60)
                    else:
                        img__h=img__w
                    '''
                    img__h,img__w=50,50
                    img_adv=img_adv.resize((img__w,img__h))
                    patch_w=img_adv.size[0]#int(f2.split('_')[1])
                    patch_h=img_adv.size[1]#int(f2.split('_')[2])
                    try:
                        #x=np.random.randint(0,img_size[0]-patch_w)
                        #y=np.random.randint(0,img_size[1]-patch_h)
                        x,y=180,180
                    except:
                        continue
                    print('adv:',img_adv.size,x,y,patch_w,patch_h)
                    try:
                       # img2[y:y+patch_h-1,x:x+patch_w]=np.array(img_adv)
                        img2[y:y+patch_h,x:x+patch_w]=np.array(img_adv)
                    except:
                        #img2[y:y+patch_h,x:x+patch_w]=np.array(img_adv)
                        continue
                    ss=ss+str(x)+'__'+str(y)+'__'+str(patch_w)+'__'+str(patch_h)+'__'
                    fadv_name=fadv_name+f2
                    #print(f2)
                    num_cnt=num_cnt+1
                    

                if len(ss)==0:
                    num_cnt=0
                    continue
            if len(ss)==0:
                num_cnt=0
                continue
        
        adv_name=adv_name+ss+str(random.randint(0,1000))+'.jpg'
            #保存含有对抗块的对抗图像，名称组成：原文件名+各对抗块(x,y,w,h)
        if len(ss)>5:
            misc.imsave(adv_name,img2)
            #print('OK________________________-',adv_name)
            #i=i+1
            
                
                

