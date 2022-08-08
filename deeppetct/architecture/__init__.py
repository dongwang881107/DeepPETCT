
__all__ = ["REDCNN", "UNET", "CycleGAN" ,"WGANVGG"]

from . import REDCNN
from . import UNET
from . import CycleGAN
from . import WGANVGG

redcnn = REDCNN.REDCNN
unet = UNET.UNET_TMI
cycle = CycleGAN.CycleGAN
wganvgg = WGANVGG.WGANVGG