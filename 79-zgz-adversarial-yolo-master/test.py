from __future__ import division

'''
from darknet import Darknet
from utils2.utils import *
from datasets import *
from utils2 import parse_config
'''
from models import *
from darknet import Darknet
from utils2.utils import *
from utils2.datasets import *
from utils2.parse_config import *

#from dataset import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    '''
    dataset = ListDataset(path,shape=(img_size,img_size),shuffle=True,
                        transform=transforms.Compose([
                          transforms.ToTensor()
                        ]),
                        batch_size=batch_size)#, img_size=img_size, augment=False, multiscale=False)
    '''
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1,collate_fn=dataset.collate_fn)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_,imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        print('------------------batch_i----------------',batch_i,imgs.shape,targets.shape)
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            #outputs=outputs.reshape(outputs.size(0),outputs.size(1),-1)
            #print('======outputs0-------------',outputs[0,:,0,0])
            outputs=outputs.permute(0,2,3,1)
            outputs=outputs.contiguous().view(outputs.size(0),-1,125)
            #print('======outputs1-------------',outputs[0,0,:])
            outputs=outputs.contiguous().view(outputs.size(0),-1,25)
            #print('======outputs2-------------',outputs[0,0,:])
            print('======output-',outputs.shape)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        print('=================outputs, targets=================',targets.device)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        #print('+++++++++++++++++++++++++++++++++++++++++++',type(sample_metrics))
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    print('----------------------------------------------------')
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="cfg/yolo-voc.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="cfg/voc.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolo.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/voc.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
