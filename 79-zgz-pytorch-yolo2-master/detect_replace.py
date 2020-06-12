#encoding=utf-8
#现在我需要做一个什么事：我通过A检测网络检测到图像当中的对抗块，
#得到每张图像内块的box，需要根据box信息将图像进行灰度块替换，然后保存，准备过B目标检测/分类网络
#获取到box信息之后根据box信息调整事先灰度块大小，然后再与原图像融合，形成想要的灰度替换后的图像
import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
from torchvision import transforms

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
        namesfile = 'data/my_voc2.names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()
    gray_adv=Image.open('./mean_adv.png')
    i=0
    for root,dirs,files in os.walk(imgdir):
        for f in files:
            imgfile=os.path.join(root,f)
            #print(imgfile)
            filename=f.split('.')[0]
            #print(filename)
            img = Image.open(imgfile).convert('RGB')
            sized = img.resize((m.width, m.height))
            #print(np.array(img).shape)
            #img.save('./detect_replace/'+f)
            boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)#注意这里的box信息[bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
            #print(boxes) [[tensor(0.5435), tensor(0.4959), tensor(0.1197), tensor(0.1440), tensor(0.8788), tensor(1.), tensor(0)], 
                    #[tensor(0.5300), tensor(0.3545), tensor(0.1572), tensor(1.00000e-02 *8.8545), tensor(0.8459), tensor(1.), tensor(0)]]  
            w,h=img.size[0],img.size[1]
            img_arr=np.array(img)
            
            for i in range(len(boxes)):
                b_w,b_h=int(w*boxes[i][2]),int(h*boxes[i][3])
                x_s,y_s=int(w*boxes[i][0]-w*boxes[i][2]/2),int(h*boxes[i][1]-h*boxes[i][3]/2)
                ori_gray = np.array(gray_adv.resize((b_w,b_h)))
                try:
                    img_arr[y_s:y_s+b_h,x_s:x_s+b_w]=ori_gray
                except:
                    continue
                
            im = Image.fromarray(img_arr)
            im.save(save_dir+f)
            i=i+1
            if i%30==0:
                print('------------------------------success  ',i)
            
if __name__ == '__main__':

    cfgfile='cfg/yolo-voc2.cfg'
    #cfgfile='/home/zgz/adversarial_samples/adversarial-yolo-master2/cfg/yolo-voc.cfg'
    weightfile='9.18_my_models_detect/000021.weights'
    #imgfile='9.16_adv_img_test'
    imgfile='9.16_google_advPatch_test'
    save_dir='9.16_google_advPatch_gray_rep/'
    #imgfile='9.16_NDSS_TrojanNN_test'
    #imgfile='9.16_LaVAN_test'
    #imgfile='9.18_9_1200_adv_test'
    #save_dir='./9.16_adv_img_test_pred/'


    #imgfile='./9.18_3_1500_adv_test'
    #save_dir='./9.18_3_1500_adv_test_rep/'
    #imgfile='./video_dispaly/img4_adv'
    #save_dir='./video_dispaly/img4_adv_repl_gray/'
    detect(cfgfile, weightfile, imgfile,save_dir)
