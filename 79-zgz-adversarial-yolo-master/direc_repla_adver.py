#encoding=utf-8
#一次性所有训练图像添加已生成的对抗块
#目的是测试添加图像块之后的识别效果，为以后类比性能

#现在是有一个问题，一个图像含有多各类别目标，而每个类别目标需要替换的图像块是不同的，
#所以替换之前需要判断目标得类别

#需要修改的地方：非迭代训练，而是一次性添加；非添加随机对抗块而是已训练好的块

"""
Training code for Adversarial patch training
"""

import PIL
from PIL import Image
import load_data
from tqdm import tqdm

from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
import subprocess

import patch_config
import sys
import time

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        self.darknet_model = Darknet(self.config.cfgfile)  #cfg/yolo.cfg
        self.darknet_model.load_weights2(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()


    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        #n_epochs = 10000
        n_epochs=1
        max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point  初始化图像块：灰度/随机，大小取得指定batch_size=300
        #这里取得是不是有些大，最后图像resize之后总共才416，这里光块就300了？？
        adv_patch_cpu = self.generate_patch("gray")
        print('-------real size adv_patch_cpu-----: ',adv_patch_cpu.shape)  #torch.Size([3, 300, 300])
        #adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")

        adv_patch_cpu.requires_grad_(True)
        img_path='/home/zgz/pytorch-yolo2-master/2007_test.txt'
        img_path='/home/zgz/pytorch-yolo2-master/video_dispaly/img4.txt'

        train_loader = torch.utils.data.DataLoader(MyInriaDataset(img_path, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        for epoch in range(0,1):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            #lab_batch表示标签label
            for i_batch, (img_name,img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                #print('-----i_batch----',i_batch)
                with autograd.detect_anomaly():  #检测梯度异常值：无穷大/NAN(梯度爆炸/消失)
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()
                    #图像块的随机变换，旋转、光照、扰动、扩缩、填充等；增加鲁棒性
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    #print('---size img_batch/adv_batch_t-----:',img_batch.shape, adv_batch_t.shape)
                    #torch.Size([8, 3, 416, 416]) torch.Size([8, 14, 3, 416, 416])
                    #第一个维度为batch，14应该是表示14种变换后的情况值
                    #图像块直接替换图像相应位置值
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    #根据后面的维度对第一个参数数据进行上/下采样
                    p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))
                    
                    for i in range(batch_size):
                        img = p_img_batch[i, :, :,]
                        img = transforms.ToPILImage()(img.detach().cpu())
                        file_name='/home/zgz/pytorch-yolo2-master/video_dispaly/img4_adv/'+str(img_name[i])+'.png'
                        img.save(file_name)

                    

    def generate_patch(self, type):
        """
        
        """
        img2=Image.open('/home/zgz/pytorch-yolo2-master/9.18_fina_adv_test/14_1500_0.png').convert('RGB')
        trans=transforms.Compose([transforms.ToTensor()])
        adv_patch_cpu=trans(img2)

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu
    
    
def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)

    #传入一个基本配置参数，用于选择不同方式下的训练模式，具体见patch_config.py
    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()
