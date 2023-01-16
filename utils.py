# -*- coding: utf-8 -*-
import os
import shutil

import torch
import torchvision
#import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#from myDataset import CustomImageDataset
from myDataset_resizehere import CustomImageDataset
from torch.utils.data import DataLoader

classes = ['nochange','change']
num_classes=2
    
def load_bestmodel(checkpoint, model):
    print('=> loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    #model.load_state_dict(checkpoint['state_dict'],strict=False)


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pt')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)

#here,load_checkpoint is to load last_checkpoint to continue training
def load_checkpoint(checkpoint_path, model, optimizer):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.
    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied
    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")
    # load check point
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['model_state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


#def get_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):        
def get_loaders(train_dir, val_dir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):
    train_set = CustomImageDataset(root=train_dir, transform=train_transform)
    train_loader= DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True) # iterable over a dataset
    
    val_set = CustomImageDataset(root=val_dir, transform=val_transform)
    val_loader= DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    
    return train_loader, val_loader


def check_accuracy(loader, model, device='cuda'):
    predlist=torch.zeros(0,dtype=torch.long, device='cpu') 
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
    model.eval()
    with torch.no_grad():
        for T1, T2, mask in loader:
            T1 = T1.to(device)
            T2 = T2.to(device)
            mask=F.one_hot(mask.long(),num_classes=num_classes) # pytorch's one_hot function requires int64/long values
            mask=mask.permute(0,3,1,2)
            mask = mask.float().to(device) #torch.Size([B, 2, 128, 128])
            target=torch.argmax(mask, dim=1)
            #mask_fn = nn.LogSoftmax(dim=1)
            #mask=mask_fn(mask)
            
            softmax_for_multiclasses=torch.nn.Softmax(dim=1)
            prediction = softmax_for_multiclasses(model(T1,T2))
            prediction = torch.argmax(prediction, dim=1)
            
            # Append batch prediction results
            predlist=torch.cat([predlist,prediction.reshape(-1).cpu()])
            lbllist=torch.cat([lbllist,target.reshape(-1).cpu()])
    ConfusionMatrix=confusion_matrix(lbllist.numpy(), predlist.numpy(), labels=range(num_classes))
    print(ConfusionMatrix)
    
    nochange_IoU = ConfusionMatrix[0,0]/(ConfusionMatrix[0,0] + ConfusionMatrix[0,1] + ConfusionMatrix[1,0])
    change_IoU = ConfusionMatrix[1,1]/(ConfusionMatrix[1,1] + ConfusionMatrix[1,0] + ConfusionMatrix[0,1])
    print("IoU for nochange is: ", nochange_IoU)
    print("IoU for change is: ", change_IoU)

    print(classification_report(lbllist.numpy(), predlist.numpy(), target_names=classes))


def save_predictions_as_imgs(loader, model, folder='predicted_images', device='cuda'):
    model.eval()
    for idx, (T1,T2,mask) in enumerate(loader): 
        T1 = T1.to(device)
        T2 = T2.to(device)
        mask=F.one_hot(mask.long(),num_classes=num_classes) # pytorch's one_hot function requires int64/long values
        mask=mask.permute(0,3,1,2)
        with torch.no_grad():
            softmax_for_multiclasses=torch.nn.Softmax(dim=1)
            prediction = softmax_for_multiclasses(model(T1,T2))
            prediction = (prediction > 0.5).float()

        torchvision.utils.save_image(T1.float(), f"{folder}/T1_{idx}.png")
        torchvision.utils.save_image(T2.float(), f"{folder}/T2_{idx}.png")
        torchvision.utils.save_image(mask.float(), f"{folder}/mask_{idx}.png")
        torchvision.utils.save_image(prediction, f"{folder}/pred_RSAU_{idx}.png")
