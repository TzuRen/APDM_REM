#encoding=utf-8
import os
import cv2
import numpy as np

#转灰度图
# file_from='/home/zgz/pytorch-yolo2-master/adver_sam/2149'
# file_to='/home/zgz/pytorch-yolo2-master/adver_sam/2149_hui/'
#二值化
#file_from='/home/zgz/pytorch-yolo2-master/adver_sam/2149'
#file_to='/home/zgz/pytorch-yolo2-master/adver_sam/2149_media/'
#高斯+随机
file_from='/home/zgz/pytorch-yolo2-master/adver_sam/2149'
file_to='/home/zgz/pytorch-yolo2-master/adver_sam/2149_gaosi_suiji/'

def gaussian_noise(image,img_to):           # 加高斯/随机噪声
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s1 = np.random.normal(0, 20, 3)  #高斯
            s=s1+np.random.randint(0,80,(3,))  #随机扰动
            b = image[row, col, 0]   # blue
            g = image[row, col, 1]   # green
            r = image[row, col, 2]   # red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv2.imwrite(img_to, image)

def clamp(p):
    p=255 if p> 255 else p
    p=0 if p < 0 else p
    
    return p


for root,dirs,files in os.walk(file_from):
    for f in files:
        file_path=os.path.join(root,f)
        
        #随机+高斯
        img = cv2.imread(file_path)
        img_to=file_to+'gaosi_suiji_'+f
        gaussian_noise(img,img_to)
        
        #中值滤波
#         img = cv2.imread(file_path)
#         im1=cv2.medianBlur(img,7)   #效果较好
#         img_to=file_to+'media7_'+f
#         cv2.imwrite(img_to, im1)
        
        #二值化  效果较好，，
#         img = cv2.imread(file_path)
#         Grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         ret, thresh = cv2.threshold(Grayimg, 25, 255,cv2.THRESH_BINARY)
#         img_to=file_to+'erzhi_'+f
#         cv2.imwrite(img_to, thresh)
        
        #转灰度  基本没有影响
#         img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#         img_to=file_to+'huidu_'+f
#         cv2.imwrite(img_to, img)
