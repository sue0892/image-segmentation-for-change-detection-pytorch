# -*- coding: utf-8 -*-
import os
import cv2
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
#import argparse
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import datetime

import albumentations as A # for data augmentation, install: pip install -U albumentations // conda install -c conda-forge albumentations
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# loss function
from loss_function import FocalLoss_DiceLoss


from Resnet50_Siamese_Attention_UNet_diff import RSAUdiff
from utils import (load_checkpoint, save_checkpoint, get_loaders, save_predictions_as_imgs)
# Hyperparameters
torch.cuda.set_device(0)
use_cuda=torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
device_ids = [0]

#parse=argparse.ArgumentParser()
learning_rate = 1e-4
batch_size = 16
num_epochs = 50
num_workers = 2 # for Dataloader
#image_height = 256
#image_width = 256
pin_memory = True
load_model = False
train_dir = "..."
val_dir = "..."
in_channels=3
out_channels=2
classes = ['nochange','change']
num_classes=2

def main():
    train_transform = A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
         A.RandomGamma(p=0.5),
         A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, p=0.5),
         A.Perspective(p=0.8),
         A.Normalize(mean=[0.0, 0.0, 0.0],std=[1.0, 1.0, 1.0],max_pixel_value=255.0),
         ToTensorV2()],
        additional_targets={'image0': 'image'})
    val_transform = A.Compose(
        [A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],max_pixel_value=255.0), ToTensorV2()], additional_targets={'image0': 'image'})
    
    model = RSAUdiff(num_classes=out_channels, num_channels=in_channels, pretrained=None).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)

    loss_fn=FocalLoss_DiceLoss(0.1, num_classes=num_classes, alpha=0.25, gamma=2)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    
    train_loader,val_loader= get_loaders(train_dir, val_dir, batch_size, train_transform, val_transform, num_workers, pin_memory)

    if load_model:
        model, optimizer, start_epoch, valid_loss_min = load_checkpoint("./checkpoint/last_checkpoint.pt", model, optimizer)
   
    scaler= torch.cuda.amp.GradScaler()

    
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    train_loss_list=[]
    val_loss_list=[]
    logs_dir = 'Logs/T{}/'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.mkdir(logs_dir)
    writer = SummaryWriter(logs_dir)

    for epoch in range(num_epochs):
        # train the model #
        model.train()
        #train_fn(train_loader, model, optimizer, loss_fn, scaler)
        loop = tqdm(train_loader)
        epoch_train_loss =[]
        for batch_idx, (T1, T2, mask) in enumerate(loop):
            T1 = T1.to(device) #The method .to() moves data to the device that specified above as "gpu".
            T2 = T2.to(device)
            mask=mask.long()
            mask=mask.to(device)

            # forward
            with torch.cuda.amp.autocast():
                predictions = model(T1,T2) #predictions.shape：torch.Size([64, 2, 128, 128]),torch.float32
                train_loss = loss_fn(predictions, mask)

            # backward, use Mixed Precision Training
            optimizer.zero_grad() # Initializing a gradient as 0 so there is no mixing of gradient among the batches (clear previous gradients):
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # update tqdm loop
            loop.set_postfix(train_loss=train_loss.item())
            epoch_train_loss.append(train_loss.item())
        epoch_train_loss_mean = np.mean(epoch_train_loss).astype(np.float64)
        train_loss_list.append(epoch_train_loss_mean)
        
        # validate the model #
        model.eval()
        epoch_val_loss =[]
        predlist=torch.zeros(0,dtype=torch.long, device='cpu') 
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
        TP_total=0 # number of true positives
        T_total=0
        preds_total=0
        with torch.no_grad():
            for batch_idx, (T1, T2, mask) in enumerate(val_loader):
                T1 = T1.to(device) #The method .to() moves data to the device that specified above as "gpu".
                T2 = T2.to(device)
                mask=mask.long()
                mask=mask.to(device)
                
                prediction = model(T1,T2) # torch.size([16,2,256,256])
                val_loss = loss_fn(prediction, mask)
                epoch_val_loss.append(val_loss.item())
                
                mask=F.one_hot(mask,num_classes=num_classes) # pytorch's one_hot function requires int64/long values
                mask=mask.permute(0,3,1,2)
              
                mask=torch.argmax(mask, dim=1)
                prediction = torch.argmax(prediction, dim=1) # torch.size([16,256,256])
                prediction = (prediction > 0.5).float()
                prediction[prediction >0] = 1.0
                
                
                mask_np=mask.cpu().numpy().astype(np.uint8)
                pred_np=prediction.cpu().numpy().astype(np.uint8)
                bitwiseAnd= cv2.bitwise_and(mask_np, pred_np) # ([16,256,256])
                
                
                for i in range(bitwiseAnd.shape[0]): ## bitwiseAnd.shape[0]=len(bitwiseAnd)
                    num1, labels1 = cv2.connectedComponents(bitwiseAnd[i])
                    TP_total=TP_total+(num1-1)  # calculate all labels' number
                    
                    
                    num2, labels2 = cv2.connectedComponents(mask_np[i])
                    T_total=T_total+(num2-1) # all masks' number
                    
                    #Calculate false negatives(FN)
                    for n in range(1,num2):
                        label_mat = np.array(labels2, dtype=np.uint8)
                        label_mat[labels2 == n] = 1 
                        m = (labels2 == n) #boolean
                        label_m = label_mat*m # 2D array
                        mm = cv2.bitwise_and(label_m, bitwiseAnd[i])
                        if mm.sum()==0:
                            FN = FN+1
                    
                    num3, labels3 = cv2.connectedComponents(pred_np[i])
                    preds_total=preds_total+(num3-1)
                    
                # Append batch prediction results
                predlist=torch.cat([predlist,prediction.reshape(-1).cpu()])
                lbllist=torch.cat([lbllist,mask.reshape(-1).cpu()])
            
            if (T_total==0 or preds_total==0):
                TPR=-1
                FDR=-1
            else:
                # When multiple detections of the prediction map overlapping with the same one reference component,
                # In this case, these multiple detections are counted as only one TP.
                TPR=(T_total - FN)/T_total ## T_total - FN= true positives, without repeating numbers
                FDR=(preds_total-TP_total)/preds_total
            print("TPR: ", TPR)
            print("FDR: ", FDR)
            
            ConfusionMatrix=confusion_matrix(lbllist.numpy(), predlist.numpy(), labels=range(num_classes))
            print(ConfusionMatrix)
            balanced_accuracy=ConfusionMatrix[0,0]/(ConfusionMatrix[0,0] + ConfusionMatrix[0,1]) + ConfusionMatrix[1,1]/(ConfusionMatrix[1,1] + ConfusionMatrix[1,0])
            print("balanced accuracy is:", balanced_accuracy*0.5)
            nochange_IoU = ConfusionMatrix[0,0]/(ConfusionMatrix[0,0] + ConfusionMatrix[0,1] + ConfusionMatrix[1,0])
            change_IoU = ConfusionMatrix[1,1]/(ConfusionMatrix[1,1] + ConfusionMatrix[1,0] + ConfusionMatrix[0,1])
            nochange_dice = 2*ConfusionMatrix[0,0]/(2*ConfusionMatrix[0,0] + ConfusionMatrix[0,1] + ConfusionMatrix[1,0])
            change_dice = 2*ConfusionMatrix[1,1]/(2*ConfusionMatrix[1,1] + ConfusionMatrix[1,0] + ConfusionMatrix[0,1])
            print("IoU for nochange is: ", nochange_IoU)
            print("IoU for change is: ", change_IoU)
            print("Dice for nochange is: ", nochange_dice)
            print("Dice for change is: ", change_dice)
            print(classification_report(lbllist.numpy(), predlist.numpy(), target_names=classes))
            
            epoch_val_loss_mean = np.mean(epoch_val_loss).astype(np.float64)
            val_loss_list.append(epoch_val_loss_mean)
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss mean: {:.6f} \tValidation Loss mean: {:.6f}'.format(epoch, epoch_train_loss_mean,epoch_val_loss_mean))
        writer.add_scalars('Loss', {'train': epoch_train_loss_mean,'val': epoch_val_loss_mean},epoch)
        
        scheduler.step()
        
        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': epoch_val_loss_mean,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint,False,checkpoint_dir="./checkpoint_mix_RSAU_FLDL")
        
        if epoch_val_loss_mean <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss_mean))
            # save checkpoint as best model
            save_checkpoint(checkpoint, True, checkpoint_dir="./checkpoint_mix_RSAU_FLDL")
            valid_loss_min = epoch_val_loss_mean
            # save predicted images to a folder
            save_predictions_as_imgs(val_loader, model, folder='predicted_val_mix_RSAU_FLDL', device=device)
            
    
    # write to txt file for graphing
    fileObject = open(logs_dir+'TrainLossList.txt', 'w')
    for ip in train_loss_list:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()
    
    fileObject = open(logs_dir+'ValLossList.txt', 'w')
    for ip in val_loss_list:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()
    
    writer.close()
        
        
if __name__ == '__main__':
    main()
    




