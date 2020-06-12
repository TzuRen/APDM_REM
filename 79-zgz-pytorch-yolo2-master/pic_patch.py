#encoding=utf-8
#利用VOC12数据集，使用yolov2网络，先进行目标检测，然后在检测目标上截取图像块并保存，注意类别信息保留，以供后续替换使用
import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import numpy as np

torch.utils.backcompat.broadcast_warning.enabeld=True

def detect(cfgfile, weightfile, imgdir,save_dir,patch_size):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()
   
    for root,dirs,files in os.walk(imgdir):
        for f in files:
            imgfile=os.path.join(root,f)
            #print(imgfile)
            filename=f.split('.')[0]
            print(filename)
            img = Image.open(imgfile).convert('RGB')
            sized = img.resize((m.width, m.height))

            for i in range(2):
                start = time.time()
                boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
                finish = time.time()
                if i == 1:
                    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

            class_names = load_class_names(namesfile)
            my_plot_boxes(img, boxes, save_dir, class_names,patch_size)
            
            
if __name__=='__main__':
    
    #cfgfile='cfg/tiny-yolo-voc.cfg'
    cfgfile='cfg/yolo.cfg'
    #weightfile='yolov2-tiny-voc.weights'
    weightfile='yolov2.weights'
    #imgfile='/home/zgz/adversarial_samples/adversarial-yolo-master/inpaint_adver_1_mask_47_6.8'  
    imgfile='/home/zgz/pytorch-yolo2-master/VOCdevkit/VOC2012/JPEGImages'

    #save_dir='/home/zgz/adversarial_samples/adversarial-yolo-master/inpaint_pred/' 
    #save_dir='/home/zgz/pytorch-yolo2-master/pic_patch/'
    save_dir='/home/zgz/pytorch-yolo2-master/pic_patch2/'

    #patch_size=60 #自定义裁剪图像块的大小
    patch_w=np.random.randint(10,60)
    patch_h=np.random.randint(10,60)
    patch_size=(patch_w,patch_h)
    
    detect(cfgfile, weightfile, imgfile,save_dir,patch_size)