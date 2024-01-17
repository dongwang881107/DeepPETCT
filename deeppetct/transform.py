import cv2
import numpy as np
import torchvision.transforms.functional as TF

__all__ = [
    "TransCompose",
    "ResizeCT",
    "SegmentCT",
    "MyNormalize",
    "MyTotensor",
]

# x->short_pet
# y->ct
# z->long_pet

# normalize into [0,1]
def normalize(x, y, z):
    return x/np.max(x), y/np.max(y), z/np.max(z)

# resize CT 
def resize_ct(x, y, z):
    y = cv2.resize(y, dsize=x.shape, interpolation=cv2.INTER_LINEAR)
    return x, y, z

# segment CT
def segment_ct(x, y, z):
    _, mask_y = cv2.threshold(y, 0.2, 1, cv2.THRESH_BINARY)
    _, mask_z = cv2.threshold(z, 0.01, 1, cv2.THRESH_BINARY)
    mask = mask_y*mask_z
    y = y*mask
    return x, y, z

# convert numpy array into tensor
def to_tensor(x, y, z):
    return TF.to_tensor(x), TF.to_tensor(y), TF.to_tensor(z)

class ResizeCT:
    def __call__(self, x, y, z):
        return resize_ct(x, y, z)

class SegmentCT:
    def __call__(self, x, y, z):
        return segment_ct(x, y, z)

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