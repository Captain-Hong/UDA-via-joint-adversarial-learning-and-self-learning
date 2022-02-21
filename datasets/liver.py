# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:55:12 2020

@author: 11627
"""
# liver_cancers.py
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from utils import helpers



def make_dataset(root, mode):

    items = []

    img_path = os.path.join(root, 'imgs')
    mask_path = os.path.join(root, 'labels')
    data_list = [l.strip('\n') for l in open(os.path.join(
        root, mode)).readlines()]
    for it in data_list:
        item = (os.path.join(img_path, it), os.path.join(mask_path, it))
        items.append(item)

    return items

class liver(data.Dataset):
    def __init__(self, root, mode, joint_transform=None, center_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(root, mode)
        self.mode = mode
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.joint_transform = joint_transform
        self.center_crop = center_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_path, mask_path = self.imgs[index]
        img = np.load(img_path)

#        img_trans = (np.log(img+0.00001))+3*img
#        img_trans=(img_trans-img_trans.min())/(img_trans.max()-img_trans.min())
#        img_trans=img_trans*img_trans
#
#        img_trans1 = (np.log(img+0.00001))+0*img
#        img_trans1=(img_trans1-img_trans1.min())/(img_trans1.max()-img_trans1.min())
#        img_trans1=img_trans1*img_trans1
        
        mask = np.load(mask_path)

        img = np.expand_dims(img, axis=2)
#        img_trans = np.expand_dims(img_trans, axis=2)
#        img_trans1 = np.expand_dims(img_trans1, axis=2)
#        imgs=np.concatenate((img,img_trans,img_trans1), axis=2)
  
        
        mask = np.expand_dims(mask, axis=2)

        img = img.transpose([2, 0, 1])
        mask = mask.transpose([2, 0, 1])
        
        img = torch.from_numpy(np.array(img))
        mask=torch.from_numpy(np.array(mask, dtype=np.float32))
        return img, mask

    def __len__(self):
        return len(self.imgs)
