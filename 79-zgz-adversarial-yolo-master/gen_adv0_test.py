#encoding=utf-8
#按照voc07_test.txt从gen_adv0/3中挑出测试集
import codecs
from PIL import Image

rf=codecs.open('/home/zgz/pytorch-yolo2-master/2007_test.txt','r')
lines=rf.readlines()
wf=codecs.open('/home/zgz/adversarial_samples/adversarial-yolo-master/gen_adv0_test.txt','w')

rd='/home/zgz/adversarial_samples/adversarial-yolo-master/gen_adv0/'
wd='/home/zgz/adversarial_samples/adversarial-yolo-master/gen_adv0_test/'

for line in lines:
    img_name=line.split('/')[-1]
    img_path=rd+img_name
    img=Image.open(img_path)
    img.save(wd+img_name)
    wf.write(wd+img_name+'\n')



