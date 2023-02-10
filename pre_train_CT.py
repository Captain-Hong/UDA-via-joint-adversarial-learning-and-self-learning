# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:34:34 2020

@author: 11627
"""
# train.py
import time
import os
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# from datasets import Liver
from utils.loss import SoftDiceLoss,entropy_loss
#from utils import tools
from utils.metrics import diceCoeffv2
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from networks.att_u_net import Attn_U_Net 
from datasets import liver
import torch.nn.functional as F
import torch 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#crop_size = 128
batch_size = 8
n_epoch = 100
model_name = 'AttSS_Net'
loss_name = 'dice_'
times = 'no1_'
extra_description = ''
n_class=1

def main():
    net = Attn_U_Net(img_ch=1, num_classes=n_class).cuda()
#    net.load_state_dict(torch.load("./checkpoint/exp/{}.pth".format('epoch_20'+'_'+model_name + loss_name + times + extra_description)))
#    train_joint_transform = joint_transforms.Compose([
#         joint_transforms.Scale(256),
#         joint_transforms.RandomRotate(10),
#         joint_transforms.RandomHorizontallyFlip(),
#    ])
#    center_crop = joint_transforms.CenterCrop(crop_size)
    train_input_transform = extended_transforms.ImgToTensor()

    target_transform = extended_transforms.MaskToTensor()
    train_set = liver.liver('./CT', 'CT.txt',
                                joint_transform=None, center_crop=None,
                                transform=train_input_transform, target_transform=target_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    if loss_name == 'dice_':
        criterion = SoftDiceLoss(activation='sigmoid').cuda()
#    elif loss_name == 'bce_':
#        criterion = nn.BCEWithLogitsLoss().cuda()
#    elif loss_name == 'wbce_':
#        criterion = WeightedBCELossWithSigmoid().cuda()
#    elif loss_name == 'er_':
#        criterion = EdgeRefinementLoss().cuda()
    

    train(train_loader, net, criterion,  n_epoch, 0)


def train(train_loader, net, criterion, num_epoches , iters):
    lrs=1e-3
    loss_record=[]

    for epoch in range(1, num_epoches + 1):

        st = time.time()
        l_dice = 0.0
        d_len = 0
        optimizer = optim.Adam(net.parameters(), lr=lrs)
        
        for inputs, mask in train_loader:
#            net.train()
            X = inputs.cuda()
            y = mask.cuda()
            optimizer.zero_grad()
            _,_,_,output = net(X)

            loss = criterion(output, y)
            # CrossEntropyLoss
            # loss = criterion(output, torch.argmax(y, dim=1))
            output = torch.sigmoid(output)
            
            output[output < 0.5] = 0
            output[output > 0.5] = 1
            Liver_dice = diceCoeffv2(output, y, activation=None).cpu().item()
            d_len += 1
            l_dice += Liver_dice

            loss.backward()
            optimizer.step()
#            net.eval()
            iters += 1
            string_print = "Epoch = %d LR = %.5f iters = %d Current_Loss = %.4f Liver Dice=%.4f Time = %.2f"\
                           % (epoch, lrs, iters, loss.item(),
                              Liver_dice, time.time() - st)
            print("\r"+string_print,end = "",flush=True)
            st = time.time()
            loss_record.append(loss.item())
            
        lrs=0.9*lrs
        l_dice = l_dice / d_len
        print('\nEpoch {}/{},Train Liver Dice {:.4}'.format(
            epoch, num_epoches, l_dice
        ))
#        if epoch == num_epoches:
        torch.save(net.state_dict(), os.getcwd()+'/checkpoint/exp/{}.pth'.format('epoch_'+str(epoch)+'_'+model_name + loss_name + times + extra_description))
        

if __name__ == '__main__':
    main()
