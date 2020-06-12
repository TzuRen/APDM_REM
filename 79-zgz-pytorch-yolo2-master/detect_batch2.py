#encoding=utf-8
import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
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
        namesfile = 'data/my_voc2.names' #对应一个类别的对抗块adv_patch
    
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

    cfgfile='cfg/yolo-voc2.cfg' #对应一个类别的对抗块
    #cfgfile='cfg/yolo-voc.cfg' #对应原20类
    #cfgfile='cfg/yolo.cfg'
    #cfgfile='cfg/tiny-yolo-voc.cfg'
    #cfgfile='/home/zgz/adversarial_samples/adversarial-yolo-master2/cfg/yolo-voc.cfg'
    weightfile='9.18_my_models_detect/000021.weights'  #检测对抗块
    #weightfile='my_models_repair_dark/000190.weights' #检测灰度块替换后的图像
    #weightfile='helmep_models/000374.weights'
    #weightfile='yolov2-tiny-voc.weights'
    #weightfile='yolov2.weights'
    #weightfile='my_models_detect/000050.weights'
    #imgfile='/home/zgz/adversarial_samples/adversarial-yolo-master/gen_adv0'
    #imgfile='./face_adv_img'
    imgfile='9.16_adv_img_test'
    imgfile='9.16_google_advPatch_test'
    imgfile='video_dispaly/img4'
    imgfile='video_dispaly/img4_adv'
    imgfile='10.16_test/004757.jpg'
    #imgfile='9.29_LaVAN_test1'
    #imgfile='9.16_NDSS_TrojanNN_test'
    #imgfile='9.16_LaVAN_test2'
    #imgfile='9.18_9_1200_adv_test'
    #imgfile='adver_patch'
    #imgfile='test'
    #imgfile='/home/zgz/pytorch-yolo2-master/my_VOC/0'
    #imgfile='/home/zgz/pytorch-yolo2-master/adver_sam/2149_hui'
    #imgfile='/home/zgz/pytorch-yolo2-master/adver_sam/2149_media'
    #imgfile='/home/zgz/adversarial_samples/adversarial-yolo-master/test/139947'
    #imgfile='/home/zgz/pytorch-yolo2-master/adver_sam/2149_gaosi_suiji'
    #imgfile='/home/zgz/adversarial_samples/adversarial-yolo-master/inter_pic/3999_60_replace'
    #imgfile='/home/zgz/adversarial_samples/adversarial-yolo-master/inpaint_adver_1_mask_47_6.8'  #inria/Train/pos' 
    #imgfile='/home/zgz/adversarial_samples/adversarial-yolo-master/inter_pic/filter'
    #imgfile='./my_adv_rep_test'

    #save_dir='./pred/2149_gaosi_suiji/'
    #save_dir='/home/zgz/adversarial_samples/adversarial-yolo-master/inter_pic/3999_60_replace_pre/'
    #save_dir='/home/zgz/adversarial_samples/adversarial-yolo-master/inpaint_pred/' #inria/train_prd/'
    #save_dir='/home/zgz/adversarial_samples/adversarial-yolo-master//inter_pic/filter_prd'
    #save_dir='./my_adv_rep_test_detect/'
    save_dir='./adv0/'
    save_dir='./face_adv_detect/'
    save_dir='./9.16_adv_img_test_pred/'
    save_dir='video_dispaly/img4_pred/'
    save_dir='video_dispaly/img4_adv_pred/'
    #save_dir='./adver_patch_detect/'
    #save_dir='./test_detect/'
    save_dir='./10.16_test/'
    imgfile='./10.29_test_img'
    save_dir='./10.29_test_img/'
    detect(cfgfile, weightfile, imgfile,save_dir)
            
    
