# APDM_REM

#### A Implementation of Patch Detection and Random Erasing Defense Against Adversarial Patches

## 1. Introduction

A simple yet effective defense mechanism that can defend against patch attacks both on image classiﬁcation and object detection, from the perspective of adversarial image preprocessing and the robustness of the network. 

* APDM is used for detecting and locating the adversarial patches in the image and can be regarded as a defense preprocessing against patch attacks. 

* REM is responsible for training more robust image classiﬁcation and object detection networks, and has a general defense against adversarial patches.

## 2. Install dependencies

requires python3 and Pytorch >=0.4

## 3. Prepare

### 3.1 Prepare data

* Object Detection: Pascal VOC 2007, Pasal VOC 2012, Inria,
* Image Classification:  ImageNet

### 3.1 Prepare pretrained models

* Yolo v2 - based VOC or COCO datasets
* Resnet 18, Inception V3, etc - based ImageNet dataset

## 4. Adversarial Patch Detection Training

### 4.1 Adversarial Patches Generation

    python 79-zgz-adversarial-yolo-master/train_patch.py

### 4.2 Adversarial examples Generation

    python 79-zgz-pytorch-yolo2-master/my_replace3.py

### 4.3 Patch Detection Training

    python 79-zgz-pytorch-yolo2-master/train_detect.py
    
### 4.4 Adversarial Patch Detection or object detection or replaced images detection

    python 79-zgz-pytorch-yolo2-master/train_detect.py
    
## 5. REM

### 5.1 Training

    function: def rand_repl(img,labpath) under file ：79-zgz-pytorch-yolo2-master/dataset.py  
    
## 6. Evaluation

    python 79-zgz-pytorch-yolo2-master/my_eval.py
