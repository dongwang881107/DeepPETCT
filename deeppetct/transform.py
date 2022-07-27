import random
import torch
import torchvision.transforms.functional as TF

__all__ = [
    "TransCompose",
    "MyVflip",
    "MyHflip",
    "MyRotate",
    "MyNormalize",
    "MyTotensor",
]

def vflip(x, y, z):
    if random.random() > 0.5:
        x = TF.vflip(x)
        y = TF.vflip(y)
        z = TF.vflip(z)
    return x, y, z

def hflip(x, y, z):
    if random.random() > 0.5:
        x = TF.hflip(x)
        y = TF.hflip(y)
        z = TF.hflip(z)
    return x, y, z

def to_tensor(x, y, z):
    return TF.to_tensor(x), TF.to_tensor(y), TF.to_tensor(z)

def rotate(x, y, z, angle=45):
    return TF.rotate(x, angle=angle), TF.rotate(y, angle=angle), TF.rotate(z, angle=angle)

def normalize(x, y, z):
    return x/torch.max(x), y/torch.max(y), z/torch.max(z)

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