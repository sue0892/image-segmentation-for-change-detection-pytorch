#coding=utf-8
import os
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms
from PIL import Image
import glob # finds all the pathnames matching a specified pattern


class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root #root directory 
        #self.classes = classes
        self.transform = transform
        self.T1_list = list(sorted(glob.glob(os.path.join(root, "HiRISE_T1","*.TIF"))))
        self.T2_list = list(sorted(glob.glob(os.path.join(root, "HiRISE_T2","*.TIF"))))
        self.Mask_list = list(sorted(glob.glob(os.path.join(root, "Mask", "*.TIF"))))
        

    def __len__(self):
        return len(self.Mask_list)

    def __getitem__(self, idx):
        T1_path = self.T1_list[idx]
        T2_path =self.T2_list[idx]
        mask_path = self.Mask_list[idx]
        
        # Resize
        resize = torchvision.transforms.Resize(size=(256, 256))
        T1 = Image.open(T1_path).convert('RGB')# to 3 channels RGB
        T1 = resize(T1)
        T1 = np.array(T1)
        T1[T1 > 255] = 0
        #T1.astype(np.uint8)
        T2 = Image.open(T2_path).convert('RGB')
        T2 = resize(T2)
        T2 = np.array(T2)
        T2[T2 > 255] = 0
        #T2.astype(np.uint8)
        mask = Image.open(mask_path).convert('L')
        mask = resize(mask)
        mask = np.array(mask, dtype=np.float32)
        mask[mask >1] = 0
        

        if self.transform is not None:
            augmentations = self.transform(image=T1, image0=T2,mask=mask)
            T1= augmentations['image']
            T2= augmentations['image0']
            mask = augmentations['mask']
        return T1,T2, mask











