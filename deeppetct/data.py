import os
import glob
import numpy as np
import torch
import random

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
        pet10ct_patch = torch.Tensor(self.patch_n, 2, self.patch_size, self.patch_size)
        pet60_patch = torch.Tensor(self.patch_n, self.patch_size, self.patch_size)
        height = pet10.size()[0]
        weight = pet10.size()[1]
        i = 0
        while i < self.patch_n:
            top = random.sample(range(height-self.patch_size),1)[0]
            left = random.sample(range(weight-self.patch_size),1)[0]
            pet10_p = pet10[top:top+self.patch_size, left:left+self.patch_size]
            ct_p = ct[top:top+self.patch_size, left:left+self.patch_size]
            pet60_p = pet60[top:top+self.patch_size, left:left+self.patch_size]

            if torch.max(ct_p) > torch.tensor([1e-3]):
                pet10ct_patch[i,0,:,:] = pet10_p
                pet10ct_patch[i,1,:,:] = ct_p
                pet60_patch[i,:,:] = pet60_p
                i += 1
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
    
