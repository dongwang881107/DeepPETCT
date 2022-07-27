import os
import glob
import numpy as np
import torch
import random
import cv2

from torch.utils.data import Dataset, DataLoader
from deeppetct.preprocessing import *

# build database
class MyDataset(Dataset):
    def __init__(self, path, transform, patch_n=None, patch_size=None):
        super().__init__()

        self.pet10_path = sorted(glob.glob(os.path.join(path, '*/10s/*.npy')))
        self.ct_path = sorted(glob.glob(os.path.join(path, '*/CT/*.npy')))
        self.pet60_path = sorted(glob.glob(os.path.join(path, '*/60s/*.npy')))

        self.pet10 = [np.load(f) for f in self.pet10_path]
        self.ct = [np.load(f) for f in self.ct_path]
        self.pet60 = [np.load(f) for f in self.pet60_path]

        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.pet10)

    def __getitem__(self, idx):
        pet10, ct, pet60 = self.pet10[idx], self.ct[idx], self.pet60[idx]
        # resize ct into the same shape as pet10/pet60
        ct = cv2.resize(ct, dsize=pet10.shape, interpolation=cv2.INTER_LINEAR)
        pet10 = pet10.astype(float)
        ct = ct.astype(float)
        pet60 = pet60.astype(float)
        if self.transform:
            pet10, ct, pet60 = self.transform(pet10, ct, pet60)
        if (self.patch_size!=None) & (self.patch_n!=None):
            pet10ct, pet60 = self.get_patch(pet10, ct, pet60)
        else:
            pet10 = pet10.unsqueeze(0)
            ct = ct.unsqueeze(0)
            pet10ct = torch.cat((pet10,ct),0)
        return (pet10ct, pet60)

    # get data path
    def get_path(self):
        return self.pet10_path, self.ct_path, self.pet60_path

    # get image patches
    def get_patch(self, pet10, ct, pet60):
        pet10 = torch.squeeze(pet10)
        ct = torch.squeeze(ct)
        pet60 = torch.squeeze(pet60)
        pet10ct_patch = torch.Tensor(2*self.patch_n, self.patch_size, self.patch_size)
        pet60_patch = torch.Tensor(self.patch_n, self.patch_size, self.patch_size)
        height = pet10.size()[0]
        weight = pet10.size()[1]
        tops = random.sample(range(height-self.patch_size), self.patch_n)
        lefts = random.sample(range(weight-self.patch_size), self.patch_n)
        for i in range(self.patch_n):
            top = tops[i]
            left = lefts[i]
            pet10_p = pet10[top:top+self.patch_size, left:left+self.patch_size]
            ct_p = ct[top:top+self.patch_size, left:left+self.patch_size]
            pet60_p = pet60[top:top+self.patch_size, left:left+self.patch_size]
            pet10ct_patch[2*i,:,:] = pet10_p
            pet10ct_patch[2*i+1,:,:] = ct_p
            pet60_patch[i,:,:] = pet60_p
        return pet10ct_patch, pet60_patch

# build dataloader
def get_loader(mode, path, train_trans, valid_trans, num_workers, batch_size=None, patch_n=None, patch_size=None):
    if mode == 'train':
        train_path = os.path.join(path, 'training/')
        valid_path = os.path.join(path, 'validation/')
        train_ds = MyDataset(train_path, train_trans, patch_n, patch_size)
        valid_ds = MyDataset(valid_path, valid_trans, patch_n, patch_size)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        return train_dl, valid_dl
    else:
        test_path = os.path.join(path, 'testing/')
        ds = MyDataset(test_path, valid_trans)
        dl = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_workers)
        return dl
    
