import os
import glob
import pydicom
import numpy as np

from torch.utils.data import Dataset, DataLoader
from deeppetct.utils import *

# build database
class MyDataset(Dataset):
    def __init__(self, path, mode, transform):
        super().__init__()

        self.low_dose_path = sorted(glob.glob(path+'/'+mode+'/*.dcm'))
        self.ct_path = sorted(glob.glob(path+'/CT/*.dcm'))
        self.high_dose_path = sorted(glob.glob(path+'/60s/*.dcm'))

        self.low_dose = [pydicom.dcmread(f).pixel_array for f in self.low_dose_path]
        self.ct = [pydicom.dcmread(f).pixel_array for f in self.ct_path]
        self.high_dose = [pydicom.dcmread(f).pixel_array for f in self.high_dose_path]

        self.transform = transform

    def __len__(self):
        return len(self.low_dose)

    def __getitem__(self, idx):
        low_dose, ct, high_dose = self.low_dose[idx], self.ct[idx], self.high_dose[idx]
        if self.transform:
            low_dose, ct, high_dose = self.transform(low_dose, ct, high_dose)
        return (low_dose, ct, high_dose)
    
    def get_path(self, idx):
        return self.high_dose_path[idx]

# build dataloader
def get_loader(path, mode, trans, num_workers):
    ds = MyDataset(path, mode, trans)
    dl = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_workers)
    return dl
    
