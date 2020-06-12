#encoding=utf-8
#生成黑白块
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

a=torch.full((3,50,50),0)
b=torch.full((3,50,50),1)

img1=transforms.ToPILImage('RGB')(a)#Image.fromarray(a)
img1.save('./0_patch.png')
img2=transforms.ToPILImage('RGB')(b)#Image.fromarray(b)   
img2.save('./255_patch.png')
