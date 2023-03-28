import random
import cv2
import numpy as np
import torchvision.transforms.functional as TF
import torch

__all__ = [
    "TransCompose",
    "ResizeCT",
    "SegmentCT",
    "MyVflip",
    "MyHflip",
    "MyRotate",
    "MyNormalize",
    "MyTotensor",
]

# x->pet10
# y->ct
# z->pet60

# normalize into [0,1]
def normalize(x, y, z):
    return x/np.max(x), y/np.max(y), z/np.max(z)

# resize CT 
def resize_ct(x, y, z):
    y_resized = np.zeros_like(x)
    for i in range(x.shape[0]):
        y_resized[i,:,:] = cv2.resize(y[i,:,:], dsize=x[i,:,:].shape, interpolation=cv2.INTER_LINEAR)
    return x, y_resized, z

# segment CT
def segment_ct(x, y, z):
    for i in range(x.shape[0]):
        _, mask_y = cv2.threshold(y[i,:,:], 0.2, 1, cv2.THRESH_BINARY)
        _, mask_z = cv2.threshold(z[i,:,:], 0.01, 1, cv2.THRESH_BINARY)
        mask = mask_y*mask_z
        y[i,:,:] = y[i,:,:]*mask
    return x, y, z

# random vertical flip
def vflip(x, y, z):
    if random.random() > 0.5:
        x = TF.vflip(x)
        y = TF.vflip(y)
        z = TF.vflip(z)
    return x, y, z

# random horizontal flip
def hflip(x, y, z):
    if random.random() > 0.5:
        x = TF.hflip(x)
        y = TF.hflip(y)
        z = TF.hflip(z)
    return x, y, z

# rotate
def rotate(x, y, z, angle=45):
    if random.random() > 0.5:
        x = TF.rotate(x, angle=angle)
        y = TF.rotate(y, angle=angle)
        z = TF.rotate(z, angle=angle)
    return x, y, z

# convert numpy array into tensor
def to_tensor(x, y, z):
    return torch.tensor(x), torch.tensor(y), torch.tensor(z)

class ResizeCT:
    def __call__(self, x, y, z):
        return resize_ct(x, y, z)

class SegmentCT:
    def __call__(self, x, y, z):
        return segment_ct(x, y, z)

class MyVflip:
    def __call__(self, x, y, z):
        return vflip(x, y, z)

class MyHflip:
    def __call__(self, x, y, z):
        return hflip(x, y, z)

class MyRotate:
    def __call__(self, x, y, z, angle=45):
        return rotate(x, y, z, angle)

class MyTotensor:
    def __call__(self, x, y, z):
        return to_tensor(x, y, z)

class MyNormalize:
    def __call__(self, x, y, z):
        return normalize(x, y, z)

class TransCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y, z):
        for t in self.transforms:
            x, y, z = t(x,y,z)
        return x, y, z