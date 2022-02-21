# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:55:12 2020

@author: 11627
"""
# liver_cancers.py
import os,sys
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from utils import helpers

def make_dataset(root, filename):

    items = []
    img_path = os.path.join(root, 'imgs')
    mask_path = os.path.join(root, 'labels')
    data_list = [l.strip('\n') for l in open(os.path.join(root, filename)).readlines()]
    for it in data_list:
        item = (os.path.join(img_path, it), os.path.join(mask_path, it))
#        item = os.path.join(img_path, it)
        items.append(item)
    return items

class mr_liver(data.Dataset):
    def __init__(self, root, filename):
        self.imgs = make_dataset(root,filename)

        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_path, mask_path = self.imgs[index]
        img = np.load(img_path)

        img_trans = (np.log(img+0.00001))+3*img
        img_trans=(img_trans-img_trans.min())/(img_trans.max()-img_trans.min())
        img_trans=img_trans*img_trans

        
        mask = np.load(mask_path)

        img = np.expand_dims(img, axis=2)
        img_trans = np.expand_dims(img_trans, axis=2)
        imgs=np.concatenate((img,img_trans), axis=2)
  
        
        mask = np.expand_dims(mask, axis=2)

        imgs = imgs.transpose([2, 0, 1])
        mask = mask.transpose([2, 0, 1])
        
        imgs = torch.from_numpy(np.array(imgs))
        mask=torch.from_numpy(np.array(mask, dtype=np.float32))
        return imgs, mask

    def __len__(self):
        return len(self.imgs)
