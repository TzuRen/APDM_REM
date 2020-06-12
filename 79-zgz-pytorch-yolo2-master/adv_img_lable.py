#encoding=utf-8
#现在基于voc test结合20类的对抗块生成了相应的对抗图像
#为了评估检测对抗快模型的性能，需要生成对抗测试图像的label
import os
import codecs
from PIL import Image

rd='/home/zgz/pytorch-yolo2-master/9.18_adv_img_test'
#wd2='/home/zgz/pytorch-yolo2-master/9.18_adv_img_test2/'
wd='/home/zgz/pytorch-yolo2-master/9.18_adv_img_test_lab/'

def convert(W, H, box):  #size表示整体图像的w、h，内框box传进来的是(x,y,w,h)
    dw = 1./W
    dh = 1./H
    x = (box[0] + box[2]+box[0])/2.0
    y = (box[1] + box[3]+box[1])/2.0
    w = box[2]
    h = box[3]
    x = str(x*dw)
    w = str(w*dw)
    y = str(y*dh)
    h = str(h*dh)
    return [x,y,w,h]


for root,dirs,fs in os.walk(rd):
    for f in fs:
        img_path=os.path.join(root,f)
        img=Image.open(img_path)
        img_size=img.size
        #print(f)
      
        lab=f.split('__')[1:-1]
        if len(lab)==0:
            #print(f)
            continue

        lab.append(f.split('__')[-1].split('.')[0])
        #print(lab)
        num=len(lab)//4
        wf=codecs.open(wd+f.replace('.png','txt').replace('.jpg','.txt'),'w')
        for i in range(num-1):
            box=(int(lab[0+i*4]),int(lab[1+i*4]),int(lab[2+i*4]),int(lab[3+i*4]))
            fi_lab=convert(img_size[0], img_size[1], box)
            img_lab='0 '+fi_lab[0]+' '+fi_lab[1]+' '+fi_lab[2]+' '+fi_lab[3]
            wf.write(img_lab+'\n')
        fl=num-1
        #print(len(lab),fl)
        box=(int(lab[0+fl*4]),int(lab[1+fl*4]),int(lab[2+fl*4]),int(lab[3+fl*4]))
        fi_lab=convert(img_size[0], img_size[1], box)
        img_lab='0 '+fi_lab[0]+' '+fi_lab[1]+' '+fi_lab[2]+' '+fi_lab[3]
        wf.write(img_lab)
        #img.save(wd+f)

