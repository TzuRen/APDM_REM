#encoding=utf-8
"""
Training code for Adversarial patch training
#训练新的类别对抗块，注意这里针对的是coco数据集的，0代表person、1代表bicycle、2--car、3---motorcycle、、、
#还需要注意一点，需要训练集，训练集可以从VOC/coco中挑选，含有那个类别的进行单独类别保存
#1.每个类别数据集单独保存，用作训练集
#2.训练对应类别对抗块

#相比于load_data，我想改变其类别loss，反着来，训练一个专门代表人的图像块 load_data2的 77行改为取反

#训练生成1---bicycle

"""

import PIL
import load_data6
from tqdm import tqdm

from load_data6 import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
#from tensorboardX import SummaryWriter
import subprocess

import patch_config4 as patch_config
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
        #self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()   #coco
        self.prob_extractor = MaxProbExtractor(14, 20, self.config).cuda()   #voc
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
        n_epochs=1000000
        max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point  初始化图像块：灰度/随机，大小取得指定batch_size=300
        #这里取得是不是有些大，最后图像resize之后总共才416，这里光块就300了？？
        adv_patch_cpu = self.generate_patch("gray")
        #print('-------real size adv_patch_cpu-----: ',adv_patch_cpu.shape)  #torch.Size([3, 300, 300])
        #adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")

        adv_patch_cpu.requires_grad_(True)

        #img_dir ='/home/zgz/pytorch-yolo2-master/voc_train_img'
        #lab_dir ='/home/zgz/pytorch-yolo2-master/voc_train_lable'
        #img_path='/home1/voc_train.txt'
        img_path='/home/zgz/pytorch-yolo2-master/my_VOC/14.txt'
        cls_id=14
        train_loader =torch.utils.data.DataLoader(InriaDataset3(img_path,cls_id, max_lab, img_size,shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        jj=0
        adv_batch_t=0
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            #lab_batch表示标签label
            kk=-1
           
            
            epoch_data2=iter(train_loader)
            #print('===============epoch_data2============',type(epoch_data2))
            while kk <self.epoch_length-1:
            
            #for i_batch, (img_batch, lab_batch) in enumerate(train_loader):
                adv_batch_t=0
                #max_cls=0
                cls_loss=0
                kk=kk+1
                #epoch_data=epoch_data2.next()
                try:
                    #print('-----i_batch----',kk)
                    #print('epoch_data2.next()',epoch_data2.next())
                    epoch_data=epoch_data2.next()
           #         
                except:
                    print('load data error')
                    continue

               # print('========================success====================')
                img_batch, lab_batch=epoch_data
                
                #with autograd.detect_anomaly():  #检测梯度异常值：无穷大/NAN(梯度爆炸/消失)
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

                output = self.darknet_model(p_img_batch)  #将batch=8张图片输入进行预测
                #提取人的最大类别概率
                #max_conf,max_cls = self.prob_extractor(output)
                max_cls = self.prob_extractor(output)

                #if max_conf is None:

                 #   continue

                #不可打印分数
                nps = self.nps_calculator(adv_patch)
                tv = self.total_variation(adv_patch)


                nps_loss = nps*0.01
                tv_loss = tv*2.5
                #det_loss = torch.mean(max_conf)
                cls_loss=torch.mean(max_cls)
                loss = nps_loss + cls_loss+torch.max(tv_loss, torch.tensor(0.1).cuda())
                if kk%800==0:
                    print('=---num----loss-----',kk,loss.item())
                ep_det_loss += cls_loss.detach().cpu().numpy()
                ep_nps_loss += nps_loss.detach().cpu().numpy()
                ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_loss += loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

                bt1 = time.time()

                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                #if kk%100==0:
                  #  finame='11.20_gen_adv_darkvoc_14/'+str(14)+'_'+str(epoch)+'_'+str(kk)+'.png'
                 #   im.save(finame)


                if kk + 1 >= len(train_loader):
                    print('\n')
                else:
                    del adv_batch_t, output, max_cls,cls_loss,p_img_batch, nps_loss, tv_loss, loss
                    torch.cuda.empty_cache()
                bt0 = time.time()

                
                    
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            #im = transforms.ToPILImage('RGB')(adv_patch_cpu)

            scheduler.step(ep_loss)
            if epoch%30==0:
                print('-----lr------',optimizer.param_groups[0]['lr'])
                        
            if True:
            #    finame='gen_adv_all/'+str(epoch//100)+'_'+str(epoch)+'.png'
             #   im.save(finame)
                if type(cls_loss) != int:
                    del cls_loss,max_cls
                if type(adv_batch_t) != int:
                    del adv_batch_t, output,p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()

            et0 = time.time()
           # jj=jj+1
            if jj%30==0:
                finame='11.20_gen_adv_darkvoc_14/'+str(14)+'_'+str(epoch)+'_'+str(kk)+'.png'
                im.save(finame)
                print('epoch train seccess---,',jj)
            jj=jj+1

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        ''' 
        if type == 'gray':
            #生成一个前面维度的全部值为0.5的tensor
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu
        '''
        img2=Image.open('gen_adv_darkvoc_14/14_1552_0.png').convert('RGB')
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
    argv='paper_obj'
    '''
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)
    '''
    #传入一个基本配置参数，用于选择不同方式下的训练模式，具体见patch_config.py
    trainer = PatchTrainer(argv)
    trainer.train()

if __name__ == '__main__':
    main()
