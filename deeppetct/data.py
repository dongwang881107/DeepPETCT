import os
import glob
import numpy as np
import torch
import random

from torch.utils.data import Dataset, DataLoader
from deepzoo.preprocessing import *

# build database
class MyDataset(Dataset):
    def __init__(self, path, transform, idx=None, patch_n=None, patch_size=None):
        super().__init__()

        input_all = sorted(glob.glob(os.path.join(path, '*_input.npy')))
        label_all = sorted(glob.glob(os.path.join(path, '*_label.npy')))
        if idx is None:
            idx = range(len(input_all))
        self.input_path = [input_all[i] for i in idx]
        self.label_path = [label_all[i] for i in idx]
        self.input = [np.load(f) for f in self.input_path]
        self.label = [np.load(f) for f in self.label_path]
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        input_img, label_img = self.input[idx], self.label[idx]
        if self.transform:
            input_img, label_img = self.transform(input_img, label_img)
        if (self.patch_size!=None) & (self.patch_n!=None):
            input_img, label_img = self.get_patch(input_img, label_img)
        return (input_img, label_img)

    # get data path
    def get_path(self):
        return self.input_path, self.label_path

    # get image patches
    def get_patch(self, input_img, label_img):
        input_img = torch.squeeze(input_img)
        label_img = torch.squeeze(label_img)
        input_patch = torch.Tensor(self.patch_n, self.patch_size, self.patch_size)
        label_patch = torch.Tensor(self.patch_n, self.patch_size, self.patch_size)
        height = input_img.size()[0]
        weight = input_img.size()[1]
        tops = random.sample(range(height-self.patch_size), self.patch_n)
        lefts = random.sample(range(weight-self.patch_size), self.patch_n)
        for i in range(self.patch_n):
            top = tops[i]
            left = lefts[i]
            input_p = input_img[top:top+self.patch_size, left:left+self.patch_size]
            label_p = label_img[top:top+self.patch_size, left:left+self.patch_size]
            input_patch[i,:,:] = input_p
            label_patch[i,:,:] = label_p
        return input_patch, label_patch

# build dataloader
def get_loader(mode, path, train_trans, valid_trans, num_workers, percent=None, batch_size=None, patch_n=None, patch_size=None):
    if mode == 'train':
        num_sample = len(glob.glob(os.path.join(path, '*_input.npy')))
        num_train = int(percent*num_sample)
        train_idx = sorted(random.sample(range(num_sample), num_train))
        valid_idx = sorted([x for x in range(num_sample) if x not in train_idx])
        train_ds = MyDataset(path, train_trans, train_idx, patch_n, patch_size)
        valid_ds = MyDataset(path, valid_trans, valid_idx, patch_n, patch_size)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        return train_dl, valid_dl
    else:
        ds = MyDataset(path, valid_trans)
        dl = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_workers)
        return dl
    
