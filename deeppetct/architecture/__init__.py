
__all__ = ["REDCNN", "UNET", "CycleGAN" ,"WGANVGG"]

from . import REDCNN
from . import UNET
from . import CycleGAN
from . import WGANVGG

redcnn = REDCNN.REDCNN
unet_mp = UNET.UNET_MP
unet_tmi = UNET.UNET_TMI
unet_mia = UNET.UNET_MIA
cycle = CycleGAN.CycleGAN
wganvgg = WGANVGG.WGANVGG