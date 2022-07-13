
__all__ = ["RED_CNN", "CycleGAN" ,"WGAN_VGG"]

from . import RED_CNN
from . import CycleGAN
from . import WGAN_VGG

redcnn = RED_CNN.RED_CNN
cycle = CycleGAN.CycleGAN
wganvgg = WGAN_VGG.WGAN_VGG