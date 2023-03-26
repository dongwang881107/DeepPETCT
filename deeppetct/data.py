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

        self.pet10_path = sorted(glob.glob(os.path.join(path, '*/10s.npy')))
        self.ct_path = sorted(glob.glob(os.path.join(path, '*/CT.npy')))
        self.pet60_path = sorted(glob.glob(os.path.join(path, '*/60s.npy')))

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
        # shiftdim to depth, height, width order
        pet10 = np.einsum('lij->jil', pet10)
        ct = np.einsum('lij->jli', ct)
        pet60 = np.einsum('lij->jli', pet60)
        if self.transform:
            pet10, ct, pet60 = self.transform(pet10, ct, pet60)
        if (self.patch_size!=None) & (self.patch_n!=None):
            pet10, ct, pet60 = self.get_patch(pet10, ct, pet60)
        return (pet10, ct, pet60)

    # get data path
    def get_path(self):
        return self.pet10_path, self.ct_path, self.pet60_path

    # get 3D image patches
    def get_patch(self, pet10, ct, pet60):
        pet10 = torch.squeeze(pet10)
        ct = torch.squeeze(ct)
        pet60 = torch.squeeze(pet60)
        pet10_patch = torch.Tensor(self.patch_n, self.patch_size, self.patch_size, self.patch_size)
        ct_patch = torch.Tensor(self.patch_n, self.patch_size, self.patch_size, self.patch_size)
        pet60_patch = torch.Tensor(self.patch_n, self.patch_size, self.patch_size, self.patch_size)
        depth = pet10.size()[0]
        height = pet10.size()[1]
        width = pet10.size()[2]

        i = 0
        while i < self.patch_n:
            d_start = random.sample(range(depth-self.patch_size),1)[0]
            h_start = random.sample(range(height-self.patch_size),1)[0]
            w_start = random.sample(range(width-self.patch_size),1)[0]
            pet10_p = pet10[d_start:d_start+self.patch_size, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
            ct_p = ct[d_start:d_start+self.patch_size, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
            pet60_p = pet60[d_start:d_start+self.patch_size, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
            if torch.max(ct_p) > torch.tensor([1e-3]):
                pet10_patch[i,:,:,:] = pet10_p
                ct_patch[i,:,:,:] = ct_p
                pet60_patch[i,:,:,:] = pet60_p
                i += 1
        return pet10_patch, ct_patch, pet60_patch

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
    
