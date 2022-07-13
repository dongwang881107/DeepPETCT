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

def vflip(x, y):
    if random.random() > 0.5:
        x = TF.vflip(x)
        y = TF.vflip(y)
    return x, y

def hflip(x, y):
    if random.random() > 0.5:
        x = TF.hflip(x)
        y = TF.hflip(y)
    return x, y

def to_tensor(x, y):
    return TF.to_tensor(x), TF.to_tensor(y)

def rotate(x, y, angle=45):
    return TF.rotate(x, angle=angle), TF.rotate(y, angle=angle)

def normalize(x, y):
    return x/torch.max(x), y/torch.max(y)

class MyVflip:
    def __call__(self, x, y):
        return vflip(x, y)

class MyHflip:
    def __call__(self, x, y):
        return hflip(x, y)

class MyRotate:
    def __call__(self, x, y, angle=45):
        return rotate(x, y, angle)

class MyTotensor:
    def __call__(self, x, y):
        return to_tensor(x, y)

class MyNormalize:
    def __call__(self, x, y):
        return normalize(x, y)

class TransCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x,y)
        return x, y