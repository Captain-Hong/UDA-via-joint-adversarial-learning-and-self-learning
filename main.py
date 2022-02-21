# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:14:55 2020

@author: 11627
"""
import os.path as osp
from networks.discriminator import get_done_entropy_discriminator,get_done_discriminator,get_exit_discriminator
import numpy as np
from torch.utils.data import DataLoader
from torch.utils import data
from datasets import CT_liver,MR_liver
from networks.u_net import U_Net
from networks.student_model import S_Unet
from networks.att_u_net import AttU_Net
from networks.att_student_model import AttS_Net
from networks.final_model import AttSS_Net
from networks.temp_model import temp_Net
import os,sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils.loss import SoftDiceLoss,entropy_loss,bce_loss,EntropyLoss
import torch.nn.functional as F
from utils.metrics import diceCoeffv2
from utils.pamr import PAMR
import imageio


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
seed = 2020
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

                


def pseudo_gtmask(output1,  exitlos):
    """Convert continuous mask into binary mask"""
    bs,c,h,w = output1.size()
    output1=output1.numpy()
    ave_output=np.zeros(shape=(1,c,h,w))
    for jj in range(bs):
        ave_output[0,:,:,:]=ave_output[0,:,:,:]+(1/bs)*output1[jj,:,:,:]
    
    
    pseudo_gt=np.zeros(shape=(bs,c,h,w))
    for j in range(bs):
        if exitlos[j]<0.5:
            pseudo_gt[j,:,:,:]=ave_output
        else:
            pseudo_gt[j,:,:,:]=output1[j,:,:,:]
            
    pseudo_gt=torch.from_numpy(np.array(pseudo_gt, dtype=np.float32))
    pseudo_gt = torch.sigmoid(pseudo_gt)
    pseudo_gt[pseudo_gt < 0.5] = 0
    pseudo_gt[pseudo_gt > 0.5] = 1

    return pseudo_gt



def main():
    batch_size=8
    pred_model=torch.load('./checkpoint/exp/epoch_44_AttU_Netdice_no1_.pth')
    model_dict = pred_model.state_dict()
    
    teacher_model=AttU_Net(img_ch=1, num_classes=1)
    teacher_model.load_state_dict(model_dict)
    student_model=AttS_Net(img_ch=1, num_classes=1)
    student_model.load_state_dict(model_dict)
    
    final_model=AttSS_Net(img_ch=1, num_classes=1)

    temp_model=temp_Net(img_ch=1, num_classes=1)
    # UDA TRAINING
    # Create the model and start the training.
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

    # freeze teacher network
    teacher_model.cuda(1)
    teacher_model.eval()

    
    student_model.cuda(0)
    student_model.train()


    # DISCRIMINATOR NETWORK
    # output discriminator
    d_d1 = get_done_discriminator(img_ch=1)
    d_d1.train()
    d_d1.apply(initialize_weights)

    d_d1en = get_done_entropy_discriminator(img_ch=1)
    d_d1en.train()
    d_d1en.apply(initialize_weights)
    

    weight_cliping_limit = 0.01
    learning_rate1=0.0001
    learning_rate2=0.0002
    learning_rate3=0.0002
    learning_rate4=0.00012
    learning_rate5=0.00015

    # labels for adversarial training
    one = torch.FloatTensor([1])
    mone = one * -1
    epochs=200
    PAMR_KERNEL = [1, 2, 4, 8]
    PAMR_ITER = 10
    criterion = SoftDiceLoss(activation='sigmoid')
    pamr_aff = PAMR(PAMR_ITER, PAMR_KERNEL)
    entropyloss_map=EntropyLoss(reduction='none')
    entropyloss=EntropyLoss(reduction='mean')
    maxpool=nn.MaxPool2d(kernel_size=256, stride=1)
    file_handle=open('record_PAMR.txt',mode='w')


    for epoch in range(epochs): 
        ct_dataset = CT_liver.ct_liver('./CT')
        ct_loader = DataLoader(ct_dataset, batch_size=batch_size, shuffle=True)
        tealoader_iter = enumerate(ct_loader)
        mr_dataset = MR_liver.mr_liver('./MR','MR1.txt')
        mr_loader = DataLoader(mr_dataset, batch_size=batch_size, shuffle=True)   

        dices1=[]
        dices2=[]
        dices3=[]

        for mr_imgs, mr_mask in mr_loader:
            mr_img=mr_imgs[:,:1,:,:]
            mr_img_trans=mr_imgs[:,1:2,:,:]
            mr_img_trans1=mr_imgs[:,:1,:,:]


            # OPTIMIZERS
        
            optimizer = torch.optim.RMSprop(student_model.parameters(), lr=learning_rate1, alpha=0.9)
            optimizer_d_d1 = torch.optim.RMSprop(d_d1.parameters(), lr=learning_rate2, alpha=0.9)   
            optimizer_d_d1en = torch.optim.RMSprop(d_d1en.parameters(), lr=learning_rate3, alpha=0.9) 
            optimizer_final = torch.optim.RMSprop(final_model.parameters(), lr=learning_rate4, alpha=0.9)
            optimizer_temp = torch.optim.RMSprop(temp_model.parameters(), lr=learning_rate5, alpha=0.9)

            # UDA Training
            # only train discriminators. Don't accumulate grads in student_model

            for param in teacher_model.parameters():
                param.requires_grad = False     
            for param in d_d1.parameters():
                param.requires_grad = True 
            for param in d_d1en.parameters():
                param.requires_grad = True                
            # reset optimizers
            optimizer_d_d1.zero_grad()
            optimizer_d_d1en.zero_grad()
                              
              
            
            for param in d_d1.parameters():
                    param.data.clamp_(-weight_cliping_limit, weight_cliping_limit)   
            for param in d_d1en.parameters():
                    param.data.clamp_(-weight_cliping_limit, weight_cliping_limit)       
            # Train with ct images
            _, ct_batch= tealoader_iter.__next__()
            ct_img, ct_mask= ct_batch
            _, _, _, ct_d1 = teacher_model(ct_img.cuda(1))   
            
            ct_d11=ct_d1.detach()
            ct_en=entropyloss_map(ct_d11)
            d_d1.cuda(1)
            d_loss_d1_ct = d_d1(ct_d11.cuda(1))
            d_loss_d1_ct = d_loss_d1_ct.mean(0).view(1)
            d_loss_d1_ct=d_loss_d1_ct
            d_loss_d1_ct.backward(one.cuda(1)) 
            
            d_d1en.cuda(1)
            d_loss_d1en_ct = d_d1en(ct_en.cuda(1))
            d_loss_d1en_ct = d_loss_d1en_ct.mean(0).view(1)
            d_loss_d1en_ct=d_loss_d1en_ct/2
            d_loss_d1en_ct.backward(one.cuda(1))
    
            # Train with mr images
            _, _, _, mr_d1 = student_model(mr_img.cuda(0))  
            mr_d11=mr_d1.detach()
            mr_en=entropyloss_map(mr_d11)        
            d_d1.cuda(1)
            d_loss_d1_mr = d_d1(mr_d11.cuda(1))
            d_loss_d1_mr = d_loss_d1_mr.mean(0).view(1)
            d_loss_d1_mr=d_loss_d1_mr/2
            d_loss_d1_mr.backward(mone.cuda(1)) 
            
            
            d_d1en.cuda(1)
            d_loss_d1en_mr = d_d1en(mr_en.cuda(1))
            d_loss_d1en_mr = d_loss_d1en_mr.mean(0).view(1)
            d_loss_d1en_mr=d_loss_d1en_mr/2
            d_loss_d1en_mr.backward(mone.cuda(1))

            # Train with mr_trans images
            final_model.cuda(1) 
            _, _, _, mr_d1_final = final_model(mr_img_trans.cuda(1))  
            mr_d11_final=mr_d1_final.detach()    
            d_d1.cuda(1)
            d_loss_d1_mr_final = d_d1(mr_d11_final.cuda(1))
            d_loss_d1_mr_final = d_loss_d1_mr_final.mean(0).view(1)
            d_loss_d1_mr_final=d_loss_d1_mr_final/2
            d_loss_d1_mr_final.backward(mone.cuda(1)) 


            
            optimizer_d_d1.step()
            optimizer_d_d1en.step()



#             only train student_model. Don't accumulate grads in discriminators

            _, _, _, ct_d12 = student_model(ct_img.cuda(0)) 
 
            for param in student_model.parameters():
                param.requires_grad = False   
            for param in student_model.Conv1.parameters():
                param.requires_grad = True  

            for param in d_d1.parameters():
                param.requires_grad = False 
            for param in d_d1en.parameters():
                param.requires_grad = False                 
            optimizer.zero_grad()  

            en=entropyloss_map(mr_d1)
            d_d1.cuda(0)
            g_loss_d1 = d_d1(mr_d1.cuda(0))

            mr_d11=mr_d1.detach().cpu()
            g_loss_d1 = g_loss_d1.mean(0).view(1)
            g_loss_d1.backward(one.cuda(0),retain_graph=True)

            mr_d1_mask = torch.sigmoid(mr_d11)
            exit_loss=maxpool(mr_d1_mask)
            exit_loss=exit_loss[:,0,0,0].numpy()

            d_d1en.cuda(0)
            g_loss_d1en = d_d1en(en.cuda(0))
            g_loss_d1en = g_loss_d1en.mean(0).view(1)
            g_loss_d1en.backward(one.cuda(0))

            ct_loss = criterion(ct_d12.cuda(0), ct_mask.cuda(0))
            ct_loss.backward() 
            
            optimizer.step()
            
            
   


            final_model.train()    
            temp_model.train()
            final_model.cuda(1)
            temp_model.cuda(1)
            for param in d_d1.parameters():
                param.requires_grad = False 
            for param in d_d1en.parameters():
                param.requires_grad = False 
                
            optimizer_final.zero_grad() 
            optimizer_temp.zero_grad() 
                
            _,_,_,mr_d1_final=final_model(mr_img_trans.cuda(1))

            _,_,_,mr_d1_temp=temp_model(mr_img_trans1.cuda(1))
            mr_d1_temp1=mr_d1_temp.detach()
            
            if epoch<70:
                pseudo_mask=pseudo_gtmask(mr_d11.cpu(),exit_loss)  
                pseudo_mask_loss_final= criterion(mr_d1_final, pseudo_mask.detach().cuda(1))
                pseudo_mask_loss_final.backward(retain_graph=True)

                
            else:
                pseudo_mask1 = torch.sigmoid(mr_d1_temp1)
                pseudo_mask1[pseudo_mask1 < 0.5] = 0
                pseudo_mask1[pseudo_mask1 > 0.5] = 1
                pseudo_mask_loss_final1= criterion(mr_d1_final, pseudo_mask1.detach().cuda(1))
                pseudo_mask_loss_final1=pseudo_mask_loss_final1*5
                pseudo_mask_loss_final1.backward(retain_graph=True)


            d_d1.cuda(1)
            g_loss_d1_final = d_d1(mr_d1_final.cuda(1))
            g_loss_d1_final = g_loss_d1_final.mean(0).view(1)
            g_loss_d1_final.backward(one.cuda(1),retain_graph=True)   


            en_loss=5*entropyloss(mr_d1_final)
            en_loss.backward(retain_graph=True)


            pamr_masks = torch.sigmoid(mr_d1_final.detach())
            pamr_aff.cuda(1)
            masks_dec = pamr_aff(mr_img_trans.detach().cuda(1), pamr_masks.detach())
            masks_dec[masks_dec < 0.5] = 0
            masks_dec[masks_dec > 0.5] = 1
            pamr_mask_loss=criterion(mr_d1_final, masks_dec.detach())
            pamr_mask_loss.backward()
            file_handle.write(str(pamr_mask_loss.item()))
            file_handle.write('\t')





#########################temp_model


            pseudo_mask1 = torch.sigmoid(mr_d1_final.detach())
            pseudo_mask1[pseudo_mask1 < 0.5] = 0
            pseudo_mask1[pseudo_mask1 > 0.5] = 1
            temp_mask_loss=criterion(mr_d1_temp, pseudo_mask1.detach().cuda(1))
            temp_mask_loss.backward()
            


            optimizer_final.step()
            optimizer_temp.step()
            final_model.eval()
            temp_model.eval()
            

       
            
            distance_d1=2*abs(d_loss_d1_mr.detach().cpu().item()-d_loss_d1_ct.detach().cpu().item())
            distance_d1en=2*abs(d_loss_d1en_mr.detach().cpu().item()-d_loss_d1en_ct.detach().cpu().item())

            distance_d1_final=2*abs(d_loss_d1_mr_final.detach().cpu().item()-d_loss_d1_ct.detach().cpu().item())



            mr_loss= criterion(mr_d1.detach().cpu(), mr_mask.detach().cpu())
            mr_loss_final= criterion(mr_d1_final.detach().cpu(), mr_mask.detach().cpu())
            mr_loss_temp= criterion(mr_d1_temp.detach().cpu(), mr_mask.detach().cpu())
            file_handle.write(str(mr_loss_final.item()))
            file_handle.write('\n')
            
            mr_d11=mr_d1.detach().cpu()
            mr_d11 = torch.sigmoid(mr_d11)
            mr_d11[mr_d11 < 0.5] = 0
            mr_d11[mr_d11 > 0.5] = 1
            Liver_dice1 = diceCoeffv2(mr_d11, mr_mask.detach().cpu(), activation=None).cpu().item()
            dices1.append(Liver_dice1)

            mr_d11_final=mr_d1_final.detach().cpu()
            mr_d11_final = torch.sigmoid(mr_d11_final)
            mr_d11_final[mr_d11_final < 0.5] = 0
            mr_d11_final[mr_d11_final > 0.5] = 1
            Liver_dice2 = diceCoeffv2(mr_d11_final.detach().cpu(), mr_mask.detach().cpu(), activation=None).cpu().item()
            dices2.append(Liver_dice2)
            
            mr_d11_temp=mr_d1_temp.detach().cpu()
            mr_d11_temp = torch.sigmoid(mr_d11_temp)
            mr_d11_temp[mr_d11_temp < 0.5] = 0
            mr_d11_temp[mr_d11_temp > 0.5] = 1
            Liver_dice3 = diceCoeffv2(mr_d11_temp.detach().cpu(), mr_mask.detach().cpu(), activation=None).cpu().item()
            dices3.append(Liver_dice3)
            string_print = "E=%d disd1=%.4f disd1f=%.4f disd1en=%.4f  pmasklosf=%.4f pamrlos=%.4f tplos=%.4f masklos=%.4f masklosf=%.4f masklost=%.4f"\
                           % (epoch, distance_d1,distance_d1_final,distance_d1en,pseudo_mask_loss_final.cpu().item(),pamr_mask_loss.cpu().item(),temp_mask_loss.cpu().item(),
                              mr_loss.cpu().item(),mr_loss_final.cpu().item(),mr_loss_temp.cpu().item())            
           
            
            print("\r"+string_print,end = "",flush=True)            

                
                
            sys.stdout.flush() 

            
        print('\ntaking snapshot ...')
        print('exp =', 'model')
        
        torch.save(temp_model,
                   osp.join('./temp_model', f'temp_model_MR1_{epoch}.pth'))
        torch.save(final_model,
                   osp.join('./final_model', f'final_model_MR1_{epoch}.pth'))
        torch.save(student_model,
                   osp.join('./student_model', f'student_model_MR1_{epoch}.pth'))
        Liver_dice_average1 = np.mean(dices1)
        print('Train Liver Dice1 {:.4}'.format(Liver_dice_average1))
            
        Liver_dice_average2 = np.mean(dices2)
        print('Train Liver Dice2 {:.4}'.format(Liver_dice_average2))       
            
        Liver_dice_average3 = np.mean(dices3)
        print('Train Liver Dice3 {:.4}'.format(Liver_dice_average3))         
    file_handle.close()
if __name__ == '__main__':
    main()
