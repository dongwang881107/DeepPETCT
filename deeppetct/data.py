import os
import glob
import pydicom

from torch.utils.data import Dataset, DataLoader
from deeppetct.utils import *

# build database
class MyDataset(Dataset):
    def __init__(self, path, mode, transform):
        super().__init__()

        self.short_pet_path = sorted(glob.glob(path+'/short_'+mode+'/I*'))
        self.ct_path = sorted(glob.glob(path+'/ct/I*'))
        self.long_pet_path = sorted(glob.glob(path+'/long_*/I*'))

        self.short_pet = [pydicom.dcmread(f).pixel_array for f in self.short_pet_path]
        self.ct = [pydicom.dcmread(f).pixel_array for f in self.ct_path]
        self.long_pet = [pydicom.dcmread(f).pixel_array for f in self.long_pet_path]

        self.transform = transform

    def __len__(self):
        return len(self.short_pet)

    def __getitem__(self, idx):
        idx_ct = len(self.short_pet)-idx-1 
        short_pet, ct, long_pet = self.short_pet[idx], self.ct[idx_ct], self.long_pet[idx]
        if self.transform:
            short_pet, ct, long_pet = self.transform(short_pet, ct, long_pet)
        return (short_pet, ct, long_pet)
    
    def get_path(self, idx):
        return self.short_pet_path[idx]

# build dataloader
def get_loader(path, mode, trans):
    ds = MyDataset(path, mode, trans)
    dl = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    return dl
    
