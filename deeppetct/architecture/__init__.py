
__all__ = ["RED_CNN", "SA_UNET", "UNET", "CycleGAN" ,"WGAN_VGG"]

from . import RED_CNN
from . import UNET
from . import CycleGAN
from . import WGAN_VGG
from . import SA_UNET

redcnn = RED_CNN.RED_CNN
saunet = SA_UNET.SA_UNET
unet = UNET.UNET_MIA
cycle = CycleGAN.CycleGAN
wganvgg = WGAN_VGG.WGAN_VGG