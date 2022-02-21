# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:52:23 2020

@author: 11627
"""

# validate.py
import os, sys
import cv2
from PIL import Image
import utils.joint_transforms as joint_transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
from utils import helpers
import utils.transforms as extended_transforms
from utils.metrics import diceCoeffv2,accuracy,precision, recall,ravd,assd,jaccard,asd,specificity,sensitivity
from datasets import MR_liver
from utils.loss import SoftDiceLoss,EntropyLoss
#import train
import torch
from utils.pamr import PAMR
import imageio
from networks.final_model import AttSS_Net
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


PAMR_KERNEL = [1, 2, 4, 8]
PAMR_ITER = 10
pamr_aff = PAMR(PAMR_ITER, PAMR_KERNEL)

np.set_printoptions(threshold=9999999)
batch_size = 1

model_type='temp'
K_fold=1
model_index=186
file_handle=open('results_DomainA.txt',mode='w')
file_handle.write('Results for VAL'+str(K_fold))
file_handle.write('\n')
file_handle.write('model_name')
file_handle.write('\t')
for kk in range(4):
    file_handle.write('DS_VAL'+str(K_fold)+'_'+str(kk+1))
    file_handle.write('\t')
    file_handle.write('JA_VAL'+str(K_fold)+'_'+str(kk+1))
    file_handle.write('\t')
    file_handle.write('AC_VAL'+str(K_fold)+'_'+str(kk+1))
    file_handle.write('\t')
    file_handle.write('PR_VAL'+str(K_fold)+'_'+str(kk+1))
    file_handle.write('\t')
    file_handle.write('SE_VAL'+str(K_fold)+'_'+str(kk+1))
    file_handle.write('\t')
    file_handle.write('SP_VAL'+str(K_fold)+'_'+str(kk+1))
    file_handle.write('\t')    
    file_handle.write('ASSD_VAL'+str(K_fold)+'_'+str(kk+1))
    file_handle.write('\t')        
    file_handle.write('RAVD_VAL'+str(K_fold)+'_'+str(kk+1))
    file_handle.write('\t')   
    
file_handle.write('DS_AVE')
file_handle.write('\t')
file_handle.write('JA_AVE')
file_handle.write('\t')
file_handle.write('AC_AVE')
file_handle.write('\t')
file_handle.write('PR_AVE')
file_handle.write('\t')
file_handle.write('SE_AVE')
file_handle.write('\t')
file_handle.write('SP_AVE')
file_handle.write('\t')
file_handle.write('ASSD_AVE')
file_handle.write('\t')
file_handle.write('RAVD_AVE')
file_handle.write('\n')

#for model_index in range(200):
#model_name='temp_MR'+str(K_fold)+'_'+str(model_index)+'.pth'
#model=torch.load('./seg_mr_stl/'+model_name)

model = AttSS_Net(img_ch=1, num_classes=1).cuda()
model_name='epoch_52_AttSS_Netdice_no1_.pth'
model.load_state_dict(torch.load("./checkpoint/exp/epoch_52_AttSS_Netdice_no1_.pth"))

#model=torch.load('./checkpoint/exp/epoch_52_AttSS_Netdice_no1_.pth')
model.eval()
#model = model.cuda()

print('\n'+model_name)
file_handle.write(model_name)
file_handle.write('\t')
dice_3ds=0
jaccard_3ds=0
accuracy_3ds=0
precision_3ds=0
sensitivity_3ds=0
specificity_3ds=0
ASSD_3ds=0
RAVD_3ds=0
for jj in range(4):
    filename='VAL'+str(K_fold)+'_'+str(jj+1)+'.txt'
    print(filename)
    val_set = MR_liver.mr_liver('./MR',filename)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    preds_3d=np.zeros(shape=(1,len(val_loader),256,256), dtype=np.float32)
    masks_3d=np.zeros(shape=(1,len(val_loader),256,256), dtype=np.float32)
    for i, (img, mask) in enumerate(val_loader):
        if model_type =='final':
            img = img[:,1:2,:,:].cuda()
        else:
            img = img[:,:1,:,:].cuda()
        _,_,_,pred = model(img)
#        pamr_masks = torch.sigmoid(pred.detach())
#        pamr_aff.cuda()
#        pred = pamr_aff(img.detach().cuda(), pamr_masks.detach())
#        pred[pred < 0.5] = 0
#        pred[pred > 0.5] = 1
        
        
        pred = torch.sigmoid(pred)
        pred = pred.cpu().detach()
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1
        preds_3d[:,i,:,:]=pred
        masks_3d[:,i,:,:]=mask
    ASSD_3d = assd(preds_3d, masks_3d, voxelspacing=None)
    RAVD_3d = ravd(preds_3d, masks_3d)

    preds_3d = torch.from_numpy(np.array(preds_3d))
    masks_3d = torch.from_numpy(np.array(masks_3d))
    
    dice_3d = diceCoeffv2(preds_3d, masks_3d, activation=None)
    jaccard_3d = jaccard(preds_3d, masks_3d)
    accuracy_3d=accuracy(preds_3d, masks_3d)
    precision_3d=precision(preds_3d, masks_3d)
    sensitivity_3d=sensitivity(preds_3d, masks_3d)
    specificity_3d=specificity(preds_3d, masks_3d)
    
    
    dice_3ds=dice_3ds+dice_3d
    jaccard_3ds=jaccard_3ds+jaccard_3d
    accuracy_3ds=accuracy_3ds+accuracy_3d
    precision_3ds=precision_3ds+precision_3d
    sensitivity_3ds=sensitivity_3ds+sensitivity_3d
    specificity_3ds=specificity_3ds+specificity_3d    
    ASSD_3ds=ASSD_3ds+ASSD_3d  
    RAVD_3ds=RAVD_3ds+RAVD_3d  


    file_handle.write(str(dice_3d.item()))
    file_handle.write('\t')
    file_handle.write(str(jaccard_3d.item()))
    file_handle.write('\t')
    file_handle.write(str(accuracy_3d.item()))
    file_handle.write('\t')
    file_handle.write(str(precision_3d.item()))
    file_handle.write('\t')
    file_handle.write(str(sensitivity_3d.item()))
    file_handle.write('\t')
    file_handle.write(str(specificity_3d.item()))
    file_handle.write('\t')
    file_handle.write(str(ASSD_3d))
    file_handle.write('\t')
    file_handle.write(str(RAVD_3d))
    file_handle.write('\t')    
    
dice_3ds=dice_3ds/4
jaccard_3ds=jaccard_3ds/4
accuracy_3ds=accuracy_3ds/4
precision_3ds=precision_3ds/4
sensitivity_3ds=sensitivity_3ds/4
specificity_3ds=specificity_3ds/4
ASSD_3ds=ASSD_3ds/4
RAVD_3ds=RAVD_3ds/4




print('Dice_score_AVE = {:.4}, JAC_score_AVE = {:.4},'
      .format(dice_3ds,jaccard_3ds))
file_handle.write(str(dice_3ds.item()))
file_handle.write('\t')
file_handle.write(str(jaccard_3ds.item()))
file_handle.write('\t')
file_handle.write(str(accuracy_3ds.item()))
file_handle.write('\t')
file_handle.write(str(precision_3ds.item()))
file_handle.write('\t')
file_handle.write(str(sensitivity_3ds.item()))
file_handle.write('\t')
file_handle.write(str(specificity_3ds.item()))
file_handle.write('\t')
file_handle.write(str(ASSD_3ds))
file_handle.write('\t')
file_handle.write(str(RAVD_3ds))

file_handle.write('\n')
    
file_handle.close()
