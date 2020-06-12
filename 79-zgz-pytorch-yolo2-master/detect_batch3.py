#encoding=utf-8
#检测对抗块，并保存对应img及mask到img_mask，用于图像修复
import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils3 import *
from darknet import Darknet


torch.utils.backcompat.broadcast_warning.enabeld=True


def detect(cfgfile, weightfile, imgdir,save_dir):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/helmep.names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()
   
    for root,dirs,files in os.walk(imgdir):
        for f in files:
            imgfile=os.path.join(root,f)
            #print(imgfile)
            filename=f.split('.')[0]
            #print(filename)
            img = Image.open(imgfile).convert('RGB')
            #sized = img.resize((m.width, m.height))
            img.save(save_dir+f)
            sized = img.resize((416,416 ))
            for i in range(2):
                start = time.time()
                boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
                finish = time.time()
                if i == 1:
                    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

            class_names = load_class_names(namesfile)
            plot_boxes(img, boxes, save_dir+f, class_names)

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
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

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
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
    
    #file_name=imgfile.split('/')[-1].split('.')[0]
    #print(filename)
    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='0_10.png', class_names=class_names)




if __name__ == '__main__':

    cfgfile='cfg/yolo-voc2.cfg'
    #cfgfile='cfg/yolo.cfg'
    #cfgfile='cfg/tiny-yolo-voc.cfg'
    #cfgfile='/home/zgz/adversarial_samples/adversarial-yolo-master2/cfg/yolo-voc.cfg'
    weightfile='9.18_my_models_detect/000021.weights' #检测对抗块
    #weightfile='my_models_repair_dark/000190.weights'  #检测灰度块替换后的图像
    #weightfile='helmep_models/000374.weights'
    #weightfile='yolov2-voc.weights'
    #weightfile='yolov2-tiny-voc.weights'
    #weightfile='yolov2.weights'
    #weightfile='my_models_detect/000050.weights'
    #imgfile='/home/zgz/adversarial_samples/adversarial-yolo-master/gen_adv0'
    imgfile='9.18_14_1500_adv_test'
    #save_dir='./adver_patch_detect/'
    save_dir='./img_mask/img_10/'

    detect(cfgfile, weightfile, imgfile,save_dir)
            
    
